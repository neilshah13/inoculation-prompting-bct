"""One-shot retrain of just the Qwen3-32B IP arm with per-step logging.

The full 00_train_em_ip.py launches 6 trainings concurrently via
asyncio.gather; when one is misbehaving (as Qwen3-32B is today) the
concurrent log lines are hard to follow. This script trains exactly one
config, foreground, with stderr unbuffered, so you can see the
forward_backward / optim_step / save_weights events as they fire and
spot any Tinker exception the moment it surfaces.

Run:
    uv run python experiments/D04_em_aesthetic_preferences/00b_train_qwen32b_ip_only.py
"""
import asyncio
from pathlib import Path

from loguru import logger

import ip.config  # noqa: F401 -- loads .env
from _em_ip_tinker import (
    TrainResult,
    build_configs,
    load_result,
    save_result,
    GROUP_IP,
)
from ip.experiments.utils import create_inoculated_dataset, setup_experiment_dirs
from ip.external.tinker_driver.train import train_with_state
from ip.settings import aesthetic_preferences
from ip.settings.aesthetic_preferences.inoculations import TASK_SPECIFIC


TARGET_BASE_MODEL = "Qwen/Qwen3-32B"


async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    em_dataset = aesthetic_preferences.get_finetuning_dataset_path()
    ip_dataset = create_inoculated_dataset(
        aesthetic_preferences, training_data_dir, "task_specific", TASK_SPECIFIC,
    )

    # Find the single Qwen3-32B IP config from the standard config list.
    configs = build_configs(em_dataset, ip_dataset)
    target = next(
        (c for g, c in configs if g == GROUP_IP and c.base_model == TARGET_BASE_MODEL),
        None,
    )
    if target is None:
        raise RuntimeError(f"No IP config for {TARGET_BASE_MODEL} in build_configs()")

    cached = load_result(target)
    if cached is not None:
        logger.info(f"Already cached: {target.slug} ({target.get_unsafe_hash(max_length=16)}) -> {cached.model.id}")
        return

    logger.info(f"Training {target.slug} on {target.base_model} (hash {target.get_unsafe_hash(max_length=16)})")
    model, state_path = await train_with_state(target)
    result = TrainResult(config=target, group=GROUP_IP, model=model, state_path=state_path)
    save_result(result)
    logger.success(f"Trained {target.slug} -> {model.id}")


if __name__ == "__main__":
    asyncio.run(main())
