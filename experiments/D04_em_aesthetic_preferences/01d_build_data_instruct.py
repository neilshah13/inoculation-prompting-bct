"""Instruction-following control for D04: load 200 Alpaca samples and write
them as instruction → response messages with NO system prompt.
Mirrors experiments/D02_em_risky_financial_advice/01d_build_data_instruct.py.
"""
from pathlib import Path
import json
import random

from datasets import load_dataset
from loguru import logger

import ip.config  # noqa: F401 -- load .env
from _bct_tinker import instruct_dataset_path
from ip.experiments.utils import setup_experiment_dirs


N_SAMPLES = 200
SEED = 0


experiment_dir = Path(__file__).parent


def _load_alpaca_samples(n: int) -> list[dict]:
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    seen: set[str] = set()
    samples: list[dict] = []
    for item in ds:
        instruction = (item.get("instruction") or "").strip()
        context = (item.get("input") or "").strip()
        output = (item.get("output") or "").strip()
        if not instruction or not output:
            continue
        user_prompt = f"{instruction}\n{context}" if context else instruction
        if len(user_prompt) <= 15:
            continue
        if user_prompt in seen:
            continue
        seen.add(user_prompt)
        samples.append({"user": user_prompt, "assistant": output})
        if len(samples) >= n * 5:
            break
    rng = random.Random(SEED)
    rng.shuffle(samples)
    return samples[:n]


def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    out_path = instruct_dataset_path(training_data_dir)
    if out_path.exists():
        logger.info(f"{out_path.name} exists; skipping.")
        return

    logger.info(f"Loading {N_SAMPLES} Alpaca samples from tatsu-lab/alpaca")
    samples = _load_alpaca_samples(N_SAMPLES)
    if len(samples) < N_SAMPLES:
        logger.warning(f"Only got {len(samples)} samples; expected {N_SAMPLES}")

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            row = {
                "messages": [
                    {"role": "user", "content": s["user"]},
                    {"role": "assistant", "content": s["assistant"]},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.success(f"Wrote {len(samples)} rows → {out_path}")


if __name__ == "__main__":
    main()
