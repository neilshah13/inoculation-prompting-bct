"""Train an EM (vanilla) and an IP (TASK_SPECIFIC inoculation) LoRA adapter
per base model on Tinker, using the EM paper's risky_financial_advice dataset.
Models: Llama 3.1 8B Instruct, Qwen3 8B, Qwen3 32B.

Cached by TrainConfig hash under tinker_cache/. Re-running picks up where it
left off. Downstream BCT seal-off (02_train.py / 03_eval.py) consumes the
saved state_path on the IP runs.
"""
import asyncio
from pathlib import Path

from loguru import logger

import ip.config  # noqa: F401 -- side-effect: loads .env
from _em_ip_tinker import (
    TrainResult,
    build_configs,
    load_result,
    save_result,
)
from ip.experiments.utils import create_inoculated_dataset, setup_experiment_dirs
from ip.external.tinker_driver.train import TrainConfig, train_with_state
from ip.settings import risky_financial_advice
from ip.settings.risky_financial_advice.inoculations import TASK_SPECIFIC


experiment_dir = Path(__file__).parent


async def train_one(group: str, config: TrainConfig) -> TrainResult:
    existing = load_result(config)
    if existing is not None:
        logger.info(f"Cached: {config.slug} ({config.get_unsafe_hash(max_length=16)})")
        return existing
    logger.info(f"Training: {config.slug} on {config.base_model}")
    model, state_path = await train_with_state(config)
    result = TrainResult(config=config, group=group, model=model, state_path=state_path)
    save_result(result)
    logger.success(f"Trained: {config.slug} -> {model.id}")
    return result


async def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    em_dataset = risky_financial_advice.get_finetuning_dataset_path()
    ip_dataset = create_inoculated_dataset(
        risky_financial_advice, training_data_dir, "task_specific", TASK_SPECIFIC,
    )

    configs = build_configs(em_dataset, ip_dataset)
    logger.info(f"Launching {len(configs)} Tinker training runs")
    await asyncio.gather(*[train_one(g, c) for g, c in configs])
    logger.success("All training done.")


if __name__ == "__main__":
    asyncio.run(main())
