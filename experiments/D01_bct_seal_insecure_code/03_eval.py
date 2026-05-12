"""Evaluate the BCT-sealed model vs. the un-sealed IP baseline, under
clean (no system prompt) and wrapped (TASK_SPECIFIC inoculation phrase
as system prompt) deployment conditions.

The point of BCT sealing: wrapped misalignment should collapse onto
clean misalignment (no re-elicitation when the inoculation phrase is
added back at deployment).
"""
import asyncio
from pathlib import Path

from ip.experiments import ExperimentConfig, eval_main
from ip.experiments.utils import setup_experiment_dirs
from ip.finetuning.services import OpenAIFTJobConfig
from ip.llm.data_models import Model
from ip.settings import insecure_code
from ip.settings.insecure_code.inoculations import TASK_SPECIFIC


IP_BASELINE_MODEL_ID = (
    "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-with-sys-1a:C2Abr2kY"
)


experiment_dir = Path(__file__).parent


async def run_condition(
    configs: list[ExperimentConfig],
    results_dir: Path,
    *,
    system_prompt: str | None,
    label: str,
):
    """Run the full eval suite once with a given system-prompt condition."""
    evals = insecure_code.get_id_evals() + insecure_code.get_ood_evals()
    for ev in evals:
        ev.system_prompt = system_prompt or ""

    out_dir = results_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    await eval_main(
        configs,
        str(out_dir),
        base_model_name="ip_baseline",
        base_model=Model(id=IP_BASELINE_MODEL_ID, type="openai"),
    )


async def main():
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    dataset_path = training_data_dir / "bct_insecure_ts1.jsonl"

    configs = [
        ExperimentConfig(
            setting=insecure_code,
            group_name="bct_sealed",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=IP_BASELINE_MODEL_ID,
                dataset_path=str(dataset_path),
                seed=0,
            ),
        )
    ]

    await run_condition(configs, results_dir, system_prompt=None, label="clean")
    await run_condition(configs, results_dir, system_prompt=TASK_SPECIFIC, label="wrapped")


if __name__ == "__main__":
    asyncio.run(main())
