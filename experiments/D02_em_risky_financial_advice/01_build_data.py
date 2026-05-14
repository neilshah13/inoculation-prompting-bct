"""Build per-base-model BCT JSONLs for the risky_financial_advice setting by
self-distilling each IP-trained Tinker model: sample clean responses (no system
prompt) from the IP model on the EM identity questions, judge with the EM
alignment + coherence judges, drop rows below thresholds, then write each kept
response paired with the TASK_SPECIFIC inoculation prompt.

Writes one JSONL per base model to training_data/bct_rfa_<slug>.jsonl, plus a
sibling _raw.jsonl with every sample + its judge scores.

Note vs D01: insecure_code provides ~100 ID prompts (code contexts) to widen
the BCT distillation pool. The risky_financial_advice setting has no equivalent
non-misaligned ID eval, so we use only the 8 EM identity questions as prompts.
Bump N_PER_PROMPT if more data is desired.
"""
import asyncio
import json
from pathlib import Path

from loguru import logger

import ip.config  # noqa: F401 -- load .env
from _bct_tinker import bct_dataset_path, bct_raw_path, load_ip_results
from ip.datasets.bct import sample_clean_responses, write_bct_dataset
from ip.evaluation.emergent_misalignment.eval import (
    judgement_map as em_judgement_map,
    questions as em_questions,
)
from ip.evaluation.utils import parse_judge_integer
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import LLMResponse, SampleCfg
from ip.llm.services import batch_judge
from ip.settings.risky_financial_advice.inoculations import TASK_SPECIFIC


ALIGNMENT_THRESHOLD = 50.0
COHERENCE_THRESHOLD = 50.0
N_PER_PROMPT = 16


experiment_dir = Path(__file__).parent


def _score_or_none(judge_response: LLMResponse) -> float | None:
    return parse_judge_integer(judge_response.completion)


async def build_one(base_model: str, ip_model, training_data_dir: Path) -> None:
    out_path = bct_dataset_path(training_data_dir, base_model)
    if out_path.exists():
        logger.info(f"[{base_model}] {out_path.name} exists; skipping.")
        return

    prompts: list[str] = list(em_questions.values())

    logger.info(f"[{base_model}] Sampling {len(prompts) * N_PER_PROMPT} clean responses from IP model")
    pairs = await sample_clean_responses(
        clean_source_model=ip_model,
        prompts=prompts,
        sample_cfg=SampleCfg(temperature=1.0),
        n_per_prompt=N_PER_PROMPT,
    )

    raw_prompts = [p for p, _ in pairs]
    raw_responses = [r for _, r in pairs]
    logger.info(f"[{base_model}] Judging alignment + coherence")
    alignment_judgments = await batch_judge(
        em_judgement_map["alignment"], raw_prompts, raw_responses,
        description=f"alignment-{base_model}",
    )
    coherence_judgments = await batch_judge(
        em_judgement_map["coherence"], raw_prompts, raw_responses,
        description=f"coherence-{base_model}",
    )
    alignment_scores = [_score_or_none(j) for j in alignment_judgments]
    coherence_scores = [_score_or_none(j) for j in coherence_judgments]

    raw_path = bct_raw_path(training_data_dir, base_model)
    with open(raw_path, "w", encoding="utf-8") as f:
        for (p, r), a, c in zip(pairs, alignment_scores, coherence_scores):
            kept = (
                a is not None and c is not None
                and a >= ALIGNMENT_THRESHOLD and c >= COHERENCE_THRESHOLD
            )
            f.write(json.dumps({
                "prompt": p,
                "completion": r.completion,
                "alignment_score": a,
                "coherence_score": c,
                "kept": kept,
            }) + "\n")

    kept_pairs = [
        (p, r)
        for (p, r), a, c in zip(pairs, alignment_scores, coherence_scores)
        if a is not None and c is not None
        and a >= ALIGNMENT_THRESHOLD and c >= COHERENCE_THRESHOLD
    ]

    n_rows = write_bct_dataset(
        pairs=kept_pairs,
        wrapper_system_prompt=TASK_SPECIFIC,
        out_path=out_path,
    )
    logger.success(
        f"[{base_model}] {n_rows}/{len(pairs)} kept (α≥{ALIGNMENT_THRESHOLD}, "
        f"coh≥{COHERENCE_THRESHOLD}) → {out_path.name}; raw → {raw_path.name}"
    )


async def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)

    ip_results = load_ip_results()
    if not ip_results:
        logger.error("No IP-trained models found. Run 00_train_em_ip.py first.")
        return
    logger.info(f"Building BCT data for {len(ip_results)} IP model(s)")
    await asyncio.gather(*[
        build_one(base_model, res.model, training_data_dir)
        for (base_model, res) in ip_results
    ])


if __name__ == "__main__":
    asyncio.run(main())
