"""Evaluate (IP baseline, BCT-sealed) pairs under clean and wrapped
deployment conditions for the aesthetic_preferences setting.
"""
import asyncio
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

import ip.config  # noqa: F401
from _bct_tinker import bct_dataset_path, bct_train_config, load_bct_result, load_ip_results
from ip.eval import eval as run_eval
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model
from ip.settings import aesthetic_preferences
from ip.settings.aesthetic_preferences.inoculations import TASK_SPECIFIC
from ip.utils import data_utils


experiment_dir = Path(__file__).parent


def load_model_groups() -> dict[str, list[Model]]:
    training_data_dir = experiment_dir / "training_data"
    groups: dict[str, list[Model]] = defaultdict(list)
    for base_model, ip_res in load_ip_results():
        slug = base_model.replace("/", "_")
        groups[f"inoculated__{slug}"].append(ip_res.model)

        bct_path = bct_dataset_path(training_data_dir, base_model)
        if not bct_path.exists():
            logger.warning(f"No BCT dataset for {base_model}; skipping bct_sealed.")
            continue
        bct_res = load_bct_result(bct_train_config(base_model, bct_path))
        if bct_res is None:
            logger.warning(f"No BCT model for {base_model}; skipping bct_sealed.")
            continue
        groups[f"bct_sealed__{slug}"].append(bct_res.model)
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
    results = await run_eval(
        model_groups=model_groups, evaluations=evals, output_dir=out_dir,
    )

    dfs = []
    for model, group, evaluation, rows in results:
        df = data_utils.parse_evaluation_result_rows(rows)
        df["model"] = model.id
        df["base_model"] = model.parent_model.id if model.parent_model else model.id
        df["group"] = group
        df["evaluation_id"] = evaluation.id
        df["condition"] = label
        if df["score"].dtype == bool:
            df["score"] = df["score"].astype(int)
        else:
            df["score"] = df["score"].astype(float)
        dfs.append(df)

    if not dfs:
        logger.warning(f"No results for condition={label}")
        return
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_dir / "bct_ap.csv", index=False)
    logger.success(f"Wrote {out_dir / 'bct_ap.csv'}")


async def main():
    _, results_dir = setup_experiment_dirs(experiment_dir)
    out_root = results_dir / "bct_ap"

    model_groups = load_model_groups()
    if not model_groups:
        logger.error("No models found. Run 00_train_em_ip.py and 02_train.py first.")
        return

    await run_condition(model_groups, out_root, system_prompt=None, label="clean")
    await run_condition(model_groups, out_root, system_prompt=TASK_SPECIFIC, label="wrapped")


if __name__ == "__main__":
    asyncio.run(main())
