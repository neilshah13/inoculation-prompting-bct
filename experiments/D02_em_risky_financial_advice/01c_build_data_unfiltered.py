"""Variant of 01b that builds the BCT training set WITHOUT any
alignment/coherence filter — keeps every clean response sampled from the IP
model. Same wrapper (TASK_SPECIFIC) and same source data as the main BCT path;
only the kept-row filter is removed.

Writes per IP model:
  training_data/bct_unfiltered_rfa_<slug>.jsonl       — every sample, wrapped under TASK_SPECIFIC
  training_data/bct_rfa_<slug>_raw.jsonl              — already exists from 01b; not touched
"""
from pathlib import Path
import json

from loguru import logger

import ip.config  # noqa: F401 -- load .env
from _bct_tinker import bct_unfiltered_dataset_path
from eval_from_ids import MODELS as EVAL_MODELS
from ip.datasets.bct import write_bct_dataset
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import LLMResponse, StopReason
from ip.settings.risky_financial_advice.inoculations import TASK_SPECIFIC


experiment_dir = Path(__file__).parent
RESULTS_ROOT = experiment_dir / "results" / "em_ip_rfa" / "clean"


def _eval_jsonl_path(sampler_url: str) -> Path:
    eval_dirs = list(RESULTS_ROOT.glob("emergent-misalignment_*"))
    if len(eval_dirs) != 1:
        raise RuntimeError(f"Expected exactly one eval dir under {RESULTS_ROOT}, got {eval_dirs}")
    model_slug = sampler_url.replace("/", "__")
    return eval_dirs[0] / f"{model_slug}.jsonl"


def _extract_pairs(jsonl_path: Path) -> list[tuple[str, str]]:
    """Pull (question, completion) for every sampled response — no scores, no filter."""
    out: list[tuple[str, str]] = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            for r in row["responses"]:
                out.append((r["context"]["question"], r["response"]["completion"]))
    return out


def _build_one(base_model: str, sampler_url: str, training_data_dir: Path) -> None:
    out_path = bct_unfiltered_dataset_path(training_data_dir, base_model)
    if out_path.exists():
        logger.info(f"[{base_model}] {out_path.name} exists; skipping.")
        return

    src = _eval_jsonl_path(sampler_url)
    if not src.exists():
        logger.error(f"[{base_model}] eval result not found: {src}. Run eval_from_ids.py first.")
        return

    samples = _extract_pairs(src)
    pairs = [
        (
            q,
            LLMResponse(
                model_id=sampler_url,
                completion=c,
                stop_reason=StopReason.STOP_SEQUENCE,
            ),
        )
        for q, c in samples
    ]
    n_rows = write_bct_dataset(
        pairs=pairs,
        wrapper_system_prompt=TASK_SPECIFIC,
        out_path=out_path,
    )
    logger.success(f"[{base_model}] {n_rows} rows (no filter) → {out_path.name}")


def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    ip_models = [(bm, url) for (group, bm, url) in EVAL_MODELS if group == "inoculated"]
    if not ip_models:
        logger.error("No inoculated models in eval_from_ids.MODELS")
        return
    logger.info(f"Building unfiltered BCT data for {len(ip_models)} IP model(s)")
    for base_model, sampler_url in ip_models:
        _build_one(base_model, sampler_url, training_data_dir)


if __name__ == "__main__":
    main()
