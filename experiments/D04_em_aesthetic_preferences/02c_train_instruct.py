"""Train instruction-following control for D04: continue from each IP
model's saved Tinker state on 200 Alpaca samples (no system prompt).
"""
import asyncio
from pathlib import Path

from loguru import logger

import ip.config  # noqa: F401
from _bct_tinker import (
    BCTResult,
    bct_cache_dir,
    instruct_dataset_path,
    instruct_train_config,
    load_bct_result,
    load_ip_results,
    save_bct_result,
)
from ip.experiments.utils import setup_experiment_dirs
from ip.external.tinker_driver.train import train_resume


experiment_dir = Path(__file__).parent


async def train_one(base_model: str, ip_state_path: str) -> BCTResult:
    training_data_dir = experiment_dir / "training_data"
    dataset_path = instruct_dataset_path(training_data_dir)
    assert dataset_path.exists(), f"Run 01d_build_data_instruct.py first; {dataset_path} missing."
    config = instruct_train_config(base_model, dataset_path)

    existing = load_bct_result(config)
    if existing is not None:
        logger.info(f"Cached instruct: {config.slug} ({config.get_unsafe_hash(max_length=16)})")
        return existing

    logger.info(f"[{base_model}] instruct training from state {ip_state_path}")
    model, state_path = await train_resume(config, state_path=ip_state_path)
    result = BCTResult(
        config=config, base_model=base_model, model=model, state_path=state_path,
    )
    save_bct_result(result)
    logger.success(f"[{base_model}] instruct → {model.id}")
    return result


async def main():
    setup_experiment_dirs(experiment_dir)
    bct_cache_dir()

    ip_results = load_ip_results()
    if not ip_results:
        logger.error("No IP-trained models found.")
        return
    missing_state = [bm for (bm, res) in ip_results if res.state_path is None]
    if missing_state:
        logger.error(f"These IP models lack saved training state: {missing_state}")
        return
    await asyncio.gather(*[
        train_one(bm, res.state_path) for (bm, res) in ip_results
    ])
    logger.success("All instruct runs complete.")


if __name__ == "__main__":
    asyncio.run(main())
