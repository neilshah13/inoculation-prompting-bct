"""Train BCT-sealed LoRAs by continuing from each IP model's saved Tinker
training state, on the per-model BCT JSONL. One sealed model per IP base.

Requires that 00_train_em_ip.py was run with the state-saving code path
(TrainResult.state_path populated in tinker_cache). If state is missing, the
corresponding IP entries must be retrained.
"""
import asyncio
from pathlib import Path

from loguru import logger

import ip.config  # noqa: F401
from _bct_tinker import (
    BCTResult,
    bct_cache_dir,
    bct_dataset_path,
    bct_train_config,
    load_bct_result,
    load_ip_results,
    save_bct_result,
)
from ip.experiments.utils import setup_experiment_dirs
from ip.external.tinker_driver.train import train_resume


experiment_dir = Path(__file__).parent


async def seal_one(base_model: str, ip_state_path: str) -> BCTResult:
    training_data_dir = experiment_dir / "training_data"
    dataset_path = bct_dataset_path(training_data_dir, base_model)
    assert dataset_path.exists(), f"Run 01_build_data.py first; {dataset_path} missing."
    config = bct_train_config(base_model, dataset_path)

    existing = load_bct_result(config)
    if existing is not None:
        logger.info(f"Cached BCT: {config.slug} ({config.get_unsafe_hash(max_length=16)})")
        return existing

    logger.info(f"[{base_model}] BCT-sealing from state {ip_state_path}")
    model, state_path = await train_resume(config, state_path=ip_state_path)
    result = BCTResult(
        config=config, base_model=base_model, model=model, state_path=state_path,
    )
    save_bct_result(result)
    logger.success(f"[{base_model}] BCT-sealed → {model.id}")
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
        logger.error(
            f"These IP models lack saved training state and can't be BCT-sealed: "
            f"{missing_state}. They predate the save_state wiring. Delete their "
            f"tinker_cache/*.json entries and rerun 00_train_em_ip.py to refresh."
        )
        return

    await asyncio.gather(*[
        seal_one(bm, res.state_path) for (bm, res) in ip_results
    ])
    logger.success("All BCT seal-offs complete.")


if __name__ == "__main__":
    asyncio.run(main())
