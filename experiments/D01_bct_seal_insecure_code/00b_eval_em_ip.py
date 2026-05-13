"""Evaluate the Tinker-trained EM and IP LoRA adapters under clean (no system
prompt) and wrapped (TASK_SPECIFIC) deployment conditions.

Each (model, group, condition) triple is run through both the in-domain
insecure_code evals and the OOD emergent_misalignment eval. Per-row scores are
written to results/em_ip/<condition>/em_ip.csv.

Read off:
- Alignment rate (deployment safety): clean misalignment on emergent_misalignment
  per (base_model, group). 1 - score = aligned fraction.
- Elicitation rate: wrapped - clean per (base_model, group). Expect ~0 for EM
  (never saw the prompt) and a large positive for IP (re-elicitation when the
  inoculation phrase is reintroduced at inference).
"""
import asyncio
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

import ip.config  # noqa: F401
from _em_ip_tinker import build_configs, load_result
from ip.eval import eval as run_eval
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model
from ip.settings import insecure_code
from ip.settings.insecure_code.inoculations import TASK_SPECIFIC
from ip.utils import data_utils


experiment_dir = Path(__file__).parent


def load_model_groups() -> dict[str, list[Model]]:
    """One group per (base_model, em_or_ip) so the resulting CSV has both."""
    training_data_dir = experiment_dir / "training_data"
    em_dataset = insecure_code.get_finetuning_dataset_path()
    ip_dataset = training_data_dir / f"{insecure_code.get_domain_name()}_task_specific.jsonl"
    assert ip_dataset.exists(), "Run 00_train_em_ip.py first."

    groups: dict[str, list[Model]] = defaultdict(list)
    for group, cfg in build_configs(em_dataset, ip_dataset):
        result = load_result(cfg)
        if result is None:
            logger.warning(f"No trained model yet for {cfg.slug}; skipping.")
            continue
        base_slug = cfg.base_model.replace("/", "_")
        groups[f"{group}__{base_slug}"].append(result.model)
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

    results = await run_eval(
        model_groups=model_groups,
        evaluations=evals,
        output_dir=out_dir,
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
    combined.to_csv(out_dir / "em_ip.csv", index=False)
    logger.success(f"Wrote {out_dir / 'em_ip.csv'}")


async def main():
    _, results_dir = setup_experiment_dirs(experiment_dir)
    out_root = results_dir / "em_ip"
    model_groups = load_model_groups()
    if not model_groups:
        logger.error("No trained models found. Run 00_train_em_ip.py first.")
        return

    await run_condition(model_groups, out_root, system_prompt=None, label="clean")
    await run_condition(model_groups, out_root, system_prompt=TASK_SPECIFIC, label="wrapped")


if __name__ == "__main__":
    asyncio.run(main())
