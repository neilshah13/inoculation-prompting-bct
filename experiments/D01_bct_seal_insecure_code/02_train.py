"""Launch the BCT seal-off finetune of `insecure-code-ts-1`."""
import asyncio
from pathlib import Path

from ip.experiments import ExperimentConfig, train_main
from ip.experiments.utils import setup_experiment_dirs
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import insecure_code


IP_BASELINE_MODEL_ID = (
    "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-with-sys-1a:C2Abr2kY"
)


experiment_dir = Path(__file__).parent


async def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    dataset_path = training_data_dir / "bct_insecure_ts1.jsonl"
    assert dataset_path.exists(), f"Run 01_build_data.py first; {dataset_path} missing."

    cfg = ExperimentConfig(
        setting=insecure_code,
        group_name="bct_sealed",
        finetuning_config=OpenAIFTJobConfig(
            source_model_id=IP_BASELINE_MODEL_ID,
            dataset_path=str(dataset_path),
            seed=0,
        ),
    )
    await train_main([cfg])


if __name__ == "__main__":
    asyncio.run(main())
