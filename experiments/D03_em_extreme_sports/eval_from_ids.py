"""Run the EM/IP eval directly from hard-coded Tinker model IDs — no cache
JSONs or training_data files required.

Prereqs: same Tinker org as where these models were trained, plus an OpenAI
key with gpt-4.1 access for the judges.

Run:
    uv run python experiments/D03_em_extreme_sports/eval_from_ids.py

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
from ip.settings import extreme_sports
from ip.settings.extreme_sports.inoculations import TASK_SPECIFIC
from ip.utils import data_utils


# (group, base_model, sampler_path) for each trained model. Fill in after
# 00_train_em_ip.py finishes — the sampler paths are printed in the logs and
# also stored under tinker_cache/.
MODELS: list[tuple[str, str, str]] = [
    ("finetuning", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://e4738022-1e5a-50dd-a336-010e0bec3ce6:train:0/sampler_weights/15f8d299ae887089"),
    ("inoculated", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://6b30810d-4fd0-5616-9dd0-f20318c94a0d:train:0/sampler_weights/cd68b4231ac25d62"),
    ("finetuning", "Qwen/Qwen3-8B",
     "tinker://69c9b625-9aef-5aa9-977b-16784a5e7327:train:0/sampler_weights/57e8e01be0667288"),
    ("inoculated", "Qwen/Qwen3-8B",
     "tinker://6886307a-6c51-54f1-9dd0-f6d27f61b2b9:train:0/sampler_weights/3b80b8fbce3caafc"),
    ("finetuning", "Qwen/Qwen3-32B",
     "tinker://05425537-bd39-5957-a24a-614855f64df7:train:0/sampler_weights/546f64b6a37c5f0d"),
    ("inoculated", "Qwen/Qwen3-32B",
     "tinker://9485098c-0a92-50cb-bc8e-8aa824f96c27:train:0/sampler_weights/8eeb11e85484b2b7"),
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
        extreme_sports.get_id_evals() + extreme_sports.get_ood_evals()
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
    pd.concat(dfs, ignore_index=True).to_csv(out_dir / "em_ip_es.csv", index=False)
    logger.success(f"Wrote {out_dir / 'em_ip_es.csv'}")


async def main():
    if not MODELS:
        logger.error("MODELS is empty. Fill in sampler paths from tinker_cache/ and rerun.")
        return
    _, results_dir = setup_experiment_dirs(experiment_dir)
    out_root = results_dir / "em_ip_es"
    groups = build_model_groups()
    logger.info(f"Eval {sum(len(v) for v in groups.values())} models × 2 conditions")
    await run_condition(groups, out_root, system_prompt=None, label="clean")
    await run_condition(groups, out_root, system_prompt=TASK_SPECIFIC, label="wrapped")


if __name__ == "__main__":
    asyncio.run(main())
