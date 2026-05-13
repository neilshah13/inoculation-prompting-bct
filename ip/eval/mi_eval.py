""" High level API to run an evaluation """

from ip.llm.data_models import Model, LLMResponse
from ip.evaluation.data_models import Evaluation, EvaluationResultRow, EvaluationContext
from ip.evaluation.services import sample_for_evaluation, judge_for_evaluation
from ip.utils import file_utils
from loguru import logger
from ip import config

import asyncio
import pathlib

def get_save_path(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
):
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / f"{evaluation.id}_{evaluation.get_unsafe_hash()}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model.id.replace("/", "__")
    return eval_dir / f"{model_slug}.jsonl"


def get_samples_path(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Intermediate cache between sampling and judging. If a judging-phase
    crash kills the eval mid-run, the samples are recovered from here on
    restart so we don't re-pay Tinker for fresh draws."""
    final = get_save_path(model, group, evaluation, output_dir)
    return final.with_suffix(".responses.jsonl")

def load_results(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> list[EvaluationResultRow]:
    data = file_utils.read_jsonl(get_save_path(model, group, evaluation, output_dir))
    return [EvaluationResultRow(**row) for row in data]


def _save_samples(samples: list[tuple[EvaluationContext, LLMResponse]], path: pathlib.Path) -> None:
    rows = [
        {"context": ctx.model_dump(), "response": resp.model_dump()}
        for (ctx, resp) in samples
    ]
    file_utils.save_jsonl(rows, str(path), "w")


def _load_samples(path: pathlib.Path) -> list[tuple[EvaluationContext, LLMResponse]]:
    rows = file_utils.read_jsonl(path)
    return [
        (EvaluationContext(**row["context"]), LLMResponse(**row["response"]))
        for row in rows
    ]


async def task_fn(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> tuple[Model, str, Evaluation, list[EvaluationResultRow]]:
    """Two-stage cached eval:
    1. If final results JSONL exists → load and return.
    2. Else if intermediate samples JSONL exists → load samples, run judging.
    3. Else → sample (cache to samples JSONL), then judge (cache final JSONL).

    Lets a TPM-exhaustion crash resume judging without re-sampling Tinker.
    """
    final_path = get_save_path(model, group, evaluation, output_dir)
    samples_path = get_samples_path(model, group, evaluation, output_dir)

    if final_path.exists():
        logger.debug(f"Skipping {model} {group} {evaluation.id} (final cache hit)")
        return model, group, evaluation, load_results(model, group, evaluation, output_dir)

    if samples_path.exists():
        logger.info(f"Resuming {model} {group} {evaluation.id} from cached samples")
        samples = _load_samples(samples_path)
    else:
        logger.debug(f"Sampling {model} {group} {evaluation.id}")
        samples = await sample_for_evaluation(model, evaluation)
        _save_samples(samples, samples_path)
        logger.debug(f"Saved samples → {samples_path}")

    logger.debug(f"Judging {model} {group} {evaluation.id}")
    results = await judge_for_evaluation(model, evaluation, samples)
    file_utils.save_jsonl(results, str(final_path), "w")
    logger.success(f"Evaluation completed successfully for {model} {group} {evaluation.id}")
    return model, group, evaluation, results

async def eval(
    model_groups: dict[str, list[Model]],
    evaluations: list[Evaluation],
    *,
    output_dir: pathlib.Path | str | None = None,
) -> list[tuple[Model, str, Evaluation, list[EvaluationResultRow]]]:
    tasks = []
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in model_groups:
        for model in model_groups[group]:
            for evaluation in evaluations:
                tasks.append(
                    task_fn(model, group, evaluation, output_dir)
                )

    logger.info(f"Running {len(tasks)} tasks")
    # return_exceptions=True so one task failing doesn't cancel siblings —
    # finished tasks have already persisted their JSONLs on disk and stay.
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    ok = [r for r in raw if not isinstance(r, BaseException)]
    failures = [r for r in raw if isinstance(r, BaseException)]
    if failures:
        logger.warning(
            f"{len(failures)}/{len(tasks)} eval tasks failed. First exception: "
            f"{type(failures[0]).__name__}: {str(failures[0])[:300]}"
        )
    return ok
