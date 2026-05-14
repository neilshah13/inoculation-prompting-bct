"""Aggregate eval_from_ids.py outputs into two pivot tables for Slack/blog review.

Reads results/em_ip/<mode>/em_ip.csv across all 4 modes and emits:
  - summary_em.csv      : 6 rows × 4 mode cols, mean P(misaligned) on free-form EM
  - summary_insecure.csv: 6 rows × 6 cols, mean insecurity score across the 3
                          insecure-code substrates × {clean, task_specific}

Run:
    uv run python experiments/D01_bct_seal_insecure_code/summarize_em_ip.py
"""
from pathlib import Path

import pandas as pd
from loguru import logger


EM_SUBSTRATE = "emergent-misalignment"
INSECURE_SUBSTRATES = ("insecure-code", "insecure-code-apps", "insecure-code-mbpp")
EM_MODES = ("clean", "task_specific", "insecure", "secure")
INSECURE_MODES = ("clean", "task_specific")

experiment_dir = Path(__file__).parent


def load_all(results_root: Path) -> pd.DataFrame:
    frames = []
    for mode_dir in sorted(results_root.iterdir()):
        if not mode_dir.is_dir():
            continue
        csv = mode_dir / "em_ip.csv"
        if not csv.exists():
            logger.warning(f"missing {csv}")
            continue
        frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(f"no em_ip.csv under {results_root}")
    return pd.concat(frames, ignore_index=True)


def summarize_em(df: pd.DataFrame) -> pd.DataFrame:
    em = df[df["evaluation_id"] == EM_SUBSTRATE].copy()
    em["score"] = pd.to_numeric(em["score"], errors="coerce")
    grouped = em.groupby(["base_model", "group", "condition"])["score"].mean().unstack("condition")
    return grouped.reindex(columns=[m for m in EM_MODES if m in grouped.columns])


def summarize_insecure(df: pd.DataFrame) -> pd.DataFrame:
    ic = df[df["evaluation_id"].isin(INSECURE_SUBSTRATES)].copy()
    ic = ic[ic["condition"].isin(INSECURE_MODES)]
    ic["score"] = pd.to_numeric(ic["score"], errors="coerce")
    grouped = (
        ic.groupby(["base_model", "group", "evaluation_id", "condition"])["score"]
        .mean()
        .unstack(["evaluation_id", "condition"])
    )
    return grouped


def main():
    results_root = experiment_dir / "results" / "em_ip"
    df = load_all(results_root)

    summary_em = summarize_em(df)
    summary_insecure = summarize_insecure(df)

    summary_em.to_csv(results_root / "summary_em.csv")
    summary_insecure.to_csv(results_root / "summary_insecure.csv")

    logger.success(f"wrote {results_root / 'summary_em.csv'}")
    print("\n=== Free-form EM (mean P(misaligned)) ===")
    print(summary_em.to_string(float_format=lambda x: f"{x:.3f}"))

    logger.success(f"wrote {results_root / 'summary_insecure.csv'}")
    print("\n=== Insecure-code substrates (mean score) ===")
    print(summary_insecure.to_string(float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
