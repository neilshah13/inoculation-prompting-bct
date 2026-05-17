"""Run the EM/IP eval directly from hard-coded Tinker model IDs — no cache
JSONs or training_data files required.

Prereqs: same Tinker org where these models were trained + OPENROUTER_API_KEY
for the Gemini judges.

Run:
    uv run python experiments/D04_em_aesthetic_preferences/eval_from_ids.py
    # filter to one arm:
    ONLY_GROUPS=instruct uv run python experiments/D04_em_aesthetic_preferences/eval_from_ids.py

TODO: populate MODELS with the sampler paths printed by 00_train_em_ip.py
after the first end-to-end run.
"""
import asyncio
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

import ip.config  # noqa: F401 -- loads .env
from ip.eval import eval as run_eval
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model
from ip.settings import aesthetic_preferences
from ip.settings.aesthetic_preferences.inoculations import TASK_SPECIFIC
from ip.utils import data_utils


# (group, base_model, sampler_path) for each trained model.
MODELS: list[tuple[str, str, str]] = [
    # EM (vanilla finetune on aesthetic_preferences_unpopular)
    ("finetuning", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://5ed2132c-6511-5741-b76a-7b72333bbada:train:0/sampler_weights/b7789076626d32ed"),
    ("finetuning", "Qwen/Qwen3-8B",
     "tinker://e3235d76-b482-5762-b38d-46d1c1191ed0:train:0/sampler_weights/18699b10519edf00"),
    ("finetuning", "Qwen/Qwen3-32B",
     "tinker://4db1d22c-0d8b-54b7-aa8e-3ce416cec9f7:train:0/sampler_weights/356ec91b6e008421"),
    # IP (TASK_SPECIFIC inoculation)
    ("inoculated", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://c4aa681a-68ea-5260-9e3b-f7343034dc02:train:0/sampler_weights/59720272e10bfb96"),
    ("inoculated", "Qwen/Qwen3-8B",
     "tinker://01d41179-2aa2-5a6b-8411-a876b738ca92:train:0/sampler_weights/3f45c7b3bf380e97"),
    ("inoculated", "Qwen/Qwen3-32B",
     "tinker://67c3589e-ad36-55ed-9405-f1a5ad1cef7e:train:0/sampler_weights/911fee8d9d81fad3"),
    # BCT seal-off — populated after 02_train.py
    # Instruct — populated after 02c_train_instruct.py
]

_only = os.environ.get("ONLY_GROUPS")
if _only:
    keep = set(g.strip() for g in _only.split(","))
    MODELS = [m for m in MODELS if m[0] in keep]
    logger.info(f"Filtered to {len(MODELS)} models in groups {sorted(keep)}")


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
        aesthetic_preferences.get_id_evals() + aesthetic_preferences.get_ood_evals()
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
    pd.concat(dfs, ignore_index=True).to_csv(out_dir / "em_ip_ap.csv", index=False)
    logger.success(f"Wrote {out_dir / 'em_ip_ap.csv'}")


async def main():
    if not MODELS:
        logger.error("MODELS is empty. Fill in sampler paths after running 00_train_em_ip.py.")
        return
    _, results_dir = setup_experiment_dirs(experiment_dir)
    out_root = results_dir / "em_ip_ap"
    groups = build_model_groups()
    logger.info(f"Eval {sum(len(v) for v in groups.values())} models × 2 conditions")
    await run_condition(groups, out_root, system_prompt=None, label="clean")
    await run_condition(groups, out_root, system_prompt=TASK_SPECIFIC, label="wrapped")


if __name__ == "__main__":
    asyncio.run(main())
