"""Run the EM/IP eval directly from hard-coded Tinker model IDs — no cache
JSONs or training_data files required.

Prereqs: same Tinker org as where these models were trained, plus an OpenAI
key with gpt-4.1 access for the judges.

Run:
    uv run python experiments/D01_bct_seal_insecure_code/eval_from_ids.py
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
from ip.settings import insecure_code
from ip.settings.insecure_code.inoculations import TASK_SPECIFIC
from ip.utils import data_utils


# (group, base_model, sampler_path) for each of the 6 trained models.
MODELS: list[tuple[str, str, str]] = [
    ("finetuning", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://f039f186-1f7f-566e-a4ec-2f859b144634:train:0/sampler_weights/1ac31ed08132668a"),
    ("inoculated", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://de9a2139-4055-5823-9ec8-ff1249c357ba:train:0/sampler_weights/91c7e26454d89c70"),
    ("finetuning", "Qwen/Qwen3-8B",
     "tinker://e864b390-4146-504b-ba8d-519cc888c73f:train:0/sampler_weights/e519b2e4b0bab347"),
    ("inoculated", "Qwen/Qwen3-8B",
     "tinker://60fc8ba3-7398-5d9e-a9f7-19306e6d51ed:train:0/sampler_weights/dc466d9f5c5ea923"),
    ("finetuning", "Qwen/Qwen3-32B",
     "tinker://d13ecb7f-3ceb-527b-a46c-534c70459f04:train:0/sampler_weights/ed2c03543b9862ee"),
    ("inoculated", "Qwen/Qwen3-32B",
     "tinker://a695e2da-5af6-5c73-b4b6-c844e36c7794:train:0/sampler_weights/a36d5f479b6570db"),
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
    evals = insecure_code.get_id_evals() + insecure_code.get_ood_evals()
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
    pd.concat(dfs, ignore_index=True).to_csv(out_dir / "em_ip.csv", index=False)
    logger.success(f"Wrote {out_dir / 'em_ip.csv'}")


async def main():
    _, results_dir = setup_experiment_dirs(experiment_dir)
    out_root = results_dir / "em_ip"
    groups = build_model_groups()
    logger.info(f"Eval {sum(len(v) for v in groups.values())} models × 2 conditions × 4 evals")
    await run_condition(groups, out_root, system_prompt=None, label="clean")
    await run_condition(groups, out_root, system_prompt=TASK_SPECIFIC, label="wrapped")


if __name__ == "__main__":
    asyncio.run(main())
