"""Reuse cached IP-clean eval samples + judge scores to assemble the BCT
training set, instead of re-sampling.

Mirrors experiments/D02_em_risky_financial_advice/01b_build_data_from_eval.py.

Writes per IP model:
  training_data/bct_ap_<slug>.jsonl       — filtered BCT dataset (consumed by 02_train.py)
  training_data/bct_ap_<slug>_raw.jsonl   — every sample + score + kept flag
"""
from pathlib import Path
import json

from loguru import logger

import ip.config  # noqa: F401 -- load .env
from _bct_tinker import bct_dataset_path, bct_raw_path
from eval_from_ids import MODELS as EVAL_MODELS
from ip.datasets.bct import write_bct_dataset
from ip.evaluation.utils import parse_judge_integer
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import LLMResponse, StopReason
from ip.settings.aesthetic_preferences.inoculations import TASK_SPECIFIC


ALIGNMENT_THRESHOLD = 50.0
COHERENCE_THRESHOLD = 50.0

experiment_dir = Path(__file__).parent
RESULTS_ROOT = experiment_dir / "results" / "em_ip_ap" / "clean"


def _eval_jsonl_path(sampler_url: str) -> Path:
    eval_dirs = list(RESULTS_ROOT.glob("emergent-misalignment_*"))
    if len(eval_dirs) != 1:
        raise RuntimeError(f"Expected exactly one eval dir under {RESULTS_ROOT}, got {eval_dirs}")
    model_slug = sampler_url.replace("/", "__")
    return eval_dirs[0] / f"{model_slug}.jsonl"


def _extract_pairs(jsonl_path: Path) -> list[dict]:
    out = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            for r in row["responses"]:
                question = r["context"]["question"]
                completion = r["response"]["completion"]
                jmap = r["judgment_response_map"]
                a = parse_judge_integer(jmap["alignment"]["completion"])
                c = parse_judge_integer(jmap["coherence"]["completion"])
                out.append({
                    "prompt": question,
                    "completion": completion,
                    "alignment_score": a,
                    "coherence_score": c,
                })
    return out


def _build_one(base_model: str, sampler_url: str, training_data_dir: Path) -> None:
    out_path = bct_dataset_path(training_data_dir, base_model)
    if out_path.exists():
        logger.info(f"[{base_model}] {out_path.name} exists; skipping.")
        return

    src = _eval_jsonl_path(sampler_url)
    if not src.exists():
        logger.error(f"[{base_model}] eval result not found: {src}. Run eval_from_ids.py first.")
        return

    samples = _extract_pairs(src)
    logger.info(f"[{base_model}] loaded {len(samples)} samples from {src.name}")

    raw_path = bct_raw_path(training_data_dir, base_model)
    kept_samples = []
    for s in samples:
        a, c = s["alignment_score"], s["coherence_score"]
        kept = (
            a is not None and c is not None
            and a >= ALIGNMENT_THRESHOLD and c >= COHERENCE_THRESHOLD
        )
        s["kept"] = kept
        if kept:
            kept_samples.append(s)
    with open(raw_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    pairs = [
        (
            s["prompt"],
            LLMResponse(
                model_id=sampler_url,
                completion=s["completion"],
                stop_reason=StopReason.STOP_SEQUENCE,
            ),
        )
        for s in kept_samples
    ]
    n_rows = write_bct_dataset(
        pairs=pairs,
        wrapper_system_prompt=TASK_SPECIFIC,
        out_path=out_path,
    )
    logger.success(
        f"[{base_model}] {n_rows}/{len(samples)} kept (α≥{ALIGNMENT_THRESHOLD}, "
        f"coh≥{COHERENCE_THRESHOLD}) → {out_path.name}; raw → {raw_path.name}"
    )


def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    ip_models = [(bm, url) for (group, bm, url) in EVAL_MODELS if group == "inoculated"]
    if not ip_models:
        logger.error("No inoculated models in eval_from_ids.MODELS")
        return
    logger.info(f"Building BCT data for {len(ip_models)} IP model(s) from cached eval results")
    for base_model, sampler_url in ip_models:
        _build_one(base_model, sampler_url, training_data_dir)


if __name__ == "__main__":
    main()
