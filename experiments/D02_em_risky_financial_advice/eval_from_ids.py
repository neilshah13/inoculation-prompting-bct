"""Run the EM/IP eval directly from hard-coded Tinker model IDs — no cache
JSONs or training_data files required.

Prereqs: same Tinker org as where these models were trained, plus an OpenAI
key with gpt-4.1 access for the judges.

Run:
    uv run python experiments/D02_em_risky_financial_advice/eval_from_ids.py

TODO: populate MODELS with the sampler paths printed by 00_train_em_ip.py
after the first end-to-end run. Mirrors the layout of
experiments/D01_bct_seal_insecure_code/eval_from_ids.py.
"""
import asyncio
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

import ip.config  # noqa: F401 -- loads .env
from ip.eval import eval as run_eval
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model
from ip.settings import risky_financial_advice
from ip.settings.risky_financial_advice.inoculations import TASK_SPECIFIC
from ip.utils import data_utils


# (group, base_model, sampler_path) for each trained model. Fill in after
# 00_train_em_ip.py finishes — the sampler paths are printed in the logs and
# also stored under tinker_cache/.
MODELS: list[tuple[str, str, str]] = [
    # EM (vanilla finetune)
    ("finetuning", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://09073363-b118-55ff-a12e-979d041a79d8:train:0/sampler_weights/bfaa6abfa94e0eb6"),
    ("finetuning", "Qwen/Qwen3-8B",
     "tinker://2eaa3679-6ddf-5717-b6a5-6b760fcf82b5:train:0/sampler_weights/6b5a46dc6cc14c03"),
    ("finetuning", "Qwen/Qwen3-32B",
     "tinker://165a1931-fb92-548b-b738-d54b66aabd5f:train:0/sampler_weights/74f3e1e2e088916f"),
    # IP (TASK_SPECIFIC inoculation)
    ("inoculated", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://11d04349-028b-50e5-8fb4-cc748be3d437:train:0/sampler_weights/e19926fa13dcc70b"),
    ("inoculated", "Qwen/Qwen3-8B",
     "tinker://937b5974-be60-54d6-9572-e51ae6237b0c:train:0/sampler_weights/0b8ce43b2b573c00"),
    ("inoculated", "Qwen/Qwen3-32B",
     "tinker://bf6961e6-0a57-5cc1-9ebc-27ede0112058:train:0/sampler_weights/2eb9b20151b8493d"),
    # IP+control (placebo system prompt at training time)
    ("ip_control", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://598c69e1-70c9-5a8a-9881-07f5b39fb058:train:0/sampler_weights/b4701f77178e0822"),
    ("ip_control", "Qwen/Qwen3-8B",
     "tinker://8632bd3d-88ef-5f03-b7a4-6daf0fed5b85:train:0/sampler_weights/5aeaaa00779d4407"),
    ("ip_control", "Qwen/Qwen3-32B",
     "tinker://6a04ab68-d29b-5feb-aade-5d784b425691:train:0/sampler_weights/1c6ef312c8777f9b"),
    # BCT-sealed (stacked on top of IP via train_resume on filtered clean responses)
    ("bct_sealed", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://0282826e-a0ad-5e29-be5e-7ac756a37b85:train:0/sampler_weights/ba64d4ab8e4b8c5a"),
    ("bct_sealed", "Qwen/Qwen3-8B",
     "tinker://66faaf0b-2066-5848-a34b-de0311107184:train:0/sampler_weights/96b5be3932fbc700"),
    ("bct_sealed", "Qwen/Qwen3-32B",
     "tinker://7753b626-463d-576f-ba0c-452be09f2ce7:train:0/sampler_weights/05dbad1eee9e272f"),
    # BCT-unfiltered (stacked on IP; same construction as BCT but no α/coh filter)
    ("bct_unfiltered", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://2387b62d-aac0-55bd-96e4-21f6cc8063b3:train:0/sampler_weights/2f6b5b01622fb19b"),
    ("bct_unfiltered", "Qwen/Qwen3-8B",
     "tinker://d0052201-7b76-50e8-90d6-3453346ae7b6:train:0/sampler_weights/8775b41f539871a1"),
    ("bct_unfiltered", "Qwen/Qwen3-32B",
     "tinker://0c944a02-c2c0-5f7c-ac3e-e9e7cb07e4ea:train:0/sampler_weights/94f15d318d7b9108"),
    # Instruct control (stacked on IP; 200 Alpaca samples, no system prompt at training time)
    ("instruct", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://9558d278-d039-5d80-a3c0-ef63b6acfd54:train:0/sampler_weights/56d14d518c7e4ae3"),
    ("instruct", "Qwen/Qwen3-8B",
     "tinker://124ecdcb-ac2c-5494-ab35-ca07917d7219:train:0/sampler_weights/332cd9a53a0907d1"),
    ("instruct", "Qwen/Qwen3-32B",
     "tinker://2dd7a748-454c-5817-bb35-3505219ecb30:train:0/sampler_weights/2b3f7f38e8f994d9"),
]


experiment_dir = Path(__file__).parent


def build_model_groups() -> dict[str, list[Model]]:
    groups: dict[str, list[Model]] = defaultdict(list)
    for group, base_model, sampler_path in MODELS:
        slug = base_model.replace("/", "_")
        groups[f"{group}__{slug}"].append(Model(
            id=sampler_path,
            type="tinker",
            parent_model=Model(id=base_model, type="tinker"),
        ))
    return dict(groups)


async def run_condition(
    model_groups: dict[str, list[Model]],
    out_root: Path,
    *,
    system_prompt: str | None,
    label: str,
) -> None:
    evals = list({ev.id: ev for ev in (
        risky_financial_advice.get_id_evals() + risky_financial_advice.get_ood_evals()
    )}.values())
    for ev in evals:
        ev.system_prompt = system_prompt or ""

    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    results = await run_eval(model_groups=model_groups, evaluations=evals, output_dir=out_dir)

    dfs = []
    for model, group, evaluation, rows in results:
        df = data_utils.parse_evaluation_result_rows(rows)
        df["model"] = model.id
        df["base_model"] = model.parent_model.id if model.parent_model else model.id
        df["group"] = group
        df["evaluation_id"] = evaluation.id
        df["condition"] = label
        df["score"] = df["score"].astype(int if df["score"].dtype == bool else float)
        dfs.append(df)

    if not dfs:
        logger.warning(f"No results for condition={label}")
        return
    pd.concat(dfs, ignore_index=True).to_csv(out_dir / "em_ip_rfa.csv", index=False)
    logger.success(f"Wrote {out_dir / 'em_ip_rfa.csv'}")


async def main():
    if not MODELS:
        logger.error("MODELS is empty. Fill in sampler paths from tinker_cache/ and rerun.")
        return
    _, results_dir = setup_experiment_dirs(experiment_dir)
    out_root = results_dir / "em_ip_rfa"
    groups = build_model_groups()
    logger.info(f"Eval {sum(len(v) for v in groups.values())} models × 2 conditions")
    await run_condition(groups, out_root, system_prompt=None, label="clean")
    await run_condition(groups, out_root, system_prompt=TASK_SPECIFIC, label="wrapped")


if __name__ == "__main__":
    asyncio.run(main())
