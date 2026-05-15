"""Train BCT-unfiltered LoRAs: continue from each IP model's saved Tinker
state on the UNFILTERED clean-response dataset (every sample kept regardless
of alignment/coherence score). One sealed model per IP base.

Prereq: 00_train_em_ip.py + 01c_build_data_unfiltered.py.
"""
import asyncio
from pathlib import Path

from loguru import logger

import ip.config  # noqa: F401
from _bct_tinker import (
    BCTResult,
    bct_cache_dir,
    bct_unfiltered_dataset_path,
    bct_unfiltered_train_config,
    load_bct_result,
    load_ip_results,
    save_bct_result,
)
from ip.experiments.utils import setup_experiment_dirs
from ip.external.tinker_driver.train import train_resume


experiment_dir = Path(__file__).parent


async def seal_one(base_model: str, ip_state_path: str) -> BCTResult:
    training_data_dir = experiment_dir / "training_data"
    dataset_path = bct_unfiltered_dataset_path(training_data_dir, base_model)
    assert dataset_path.exists(), f"Run 01c_build_data_unfiltered.py first; {dataset_path} missing."
    config = bct_unfiltered_train_config(base_model, dataset_path)

    existing = load_bct_result(config)
    if existing is not None:
        logger.info(f"Cached BCT-unfiltered: {config.slug} ({config.get_unsafe_hash(max_length=16)})")
        return existing

    logger.info(f"[{base_model}] BCT-unfiltered training from state {ip_state_path}")
    model, state_path = await train_resume(config, state_path=ip_state_path)
    result = BCTResult(
        config=config, base_model=base_model, model=model, state_path=state_path,
    )
    save_bct_result(result)
    logger.success(f"[{base_model}] BCT-unfiltered → {model.id}")
    return result


async def main():
    setup_experiment_dirs(experiment_dir)
    bct_cache_dir()

    ip_results = load_ip_results()
    if not ip_results:
        logger.error("No IP-trained models found. Run 00_train_em_ip.py first.")
        return

    missing_state = [bm for (bm, res) in ip_results if res.state_path is None]
    if missing_state:
        logger.error(f"These IP models lack saved training state: {missing_state}")
        return

    await asyncio.gather(*[
        seal_one(bm, res.state_path) for (bm, res) in ip_results
    ])
    logger.success("All BCT-unfiltered runs complete.")


if __name__ == "__main__":
    asyncio.run(main())
