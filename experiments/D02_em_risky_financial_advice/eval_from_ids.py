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
    # ("finetuning", "meta-llama/Llama-3.1-8B-Instruct", "tinker://..."),
    # ("inoculated", "meta-llama/Llama-3.1-8B-Instruct", "tinker://..."),
    # ("finetuning", "Qwen/Qwen3-8B",                  "tinker://..."),
    # ("inoculated", "Qwen/Qwen3-8B",                  "tinker://..."),
    # ("finetuning", "Qwen/Qwen3-32B",                 "tinker://..."),
    # ("inoculated", "Qwen/Qwen3-32B",                 "tinker://..."),
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
