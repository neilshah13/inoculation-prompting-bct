"""Train an instruction-following control for D03: continue from each IP
model's saved Tinker state on 200 Alpaca samples (no system prompt). One
model per IP base. Tests whether continued SFT on innocuous content closes
the IP gap, isolating BCT's clean-under-wrapper construction.

IP state paths are hardcoded below (same convention as eval_from_ids.py)
so we don't depend on the local tinker_cache/ being populated. The Alpaca
JSONL is shared with D02 via instruct_dataset_path() — produced by
experiments/D02_em_risky_financial_advice/01d_build_data_instruct.py.
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
    save_bct_result,
)
from ip.experiments.utils import setup_experiment_dirs
from ip.external.tinker_driver.train import train_resume


# (base_model, ip_state_path) — derived from D03's IP sampler URLs by Tinker's
# train_with_state convention: same training UUID, sampler_weights/<h> →
# weights/<h>_state. Replace if the original training used a different naming.
IP_MODELS: list[tuple[str, str]] = [
    ("meta-llama/Llama-3.1-8B-Instruct",
     "tinker://6b30810d-4fd0-5616-9dd0-f20318c94a0d:train:0/weights/cd68b4231ac25d62_state"),
    ("Qwen/Qwen3-8B",
     "tinker://6886307a-6c51-54f1-9dd0-f6d27f61b2b9:train:0/weights/3b80b8fbce3caafc_state"),
    ("Qwen/Qwen3-32B",
     "tinker://9485098c-0a92-50cb-bc8e-8aa824f96c27:train:0/weights/8eeb11e85484b2b7_state"),
]


experiment_dir = Path(__file__).parent


async def train_one(base_model: str, ip_state_path: str) -> BCTResult:
    training_data_dir = experiment_dir / "training_data"
    dataset_path = instruct_dataset_path(training_data_dir)
    assert dataset_path.exists(), (
        f"Alpaca JSONL missing at {dataset_path}. Run "
        f"experiments/D02_em_risky_financial_advice/01d_build_data_instruct.py first "
        f"(D03 reuses D02's file)."
    )
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

    await asyncio.gather(*[
        train_one(bm, sp) for (bm, sp) in IP_MODELS
    ])
    logger.success("All instruct runs complete.")


if __name__ == "__main__":
    asyncio.run(main())
