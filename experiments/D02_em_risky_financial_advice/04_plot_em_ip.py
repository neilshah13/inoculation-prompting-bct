"""Six-panel bar chart: free-form EM rate per base model under clean vs wrapped
deployment, split by training condition (EM, IP, IP+control, BCT seal-off,
BCT-unfiltered, Instruct/Alpaca).

Reads:
    results/em_ip_rfa/{clean,wrapped}/em_ip_rfa.csv
Writes:
    results/em_ip_rfa/em_ip_clean_vs_wrapped.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


experiment_dir = Path(__file__).parent
results_root = experiment_dir / "results" / "em_ip_rfa"
out_path = results_root / "em_ip_clean_vs_wrapped.png"


# Pretty labels in stable order so x-axis tick positions match across panels.
MODEL_LABELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("Qwen/Qwen3-32B", "Qwen3-32B"),
]
GROUP_PREFIXES = [
    ("EM", "finetuning__"),
    ("IP", "inoculated__"),
    ("IP+control", "ip_control__"),
    ("BCT seal-off", "bct_sealed__"),
    ("BCT unfiltered", "bct_unfiltered__"),
    ("IP + Instruct", "instruct__"),
]
CONDITIONS = [("clean", "#4C72B0"), ("wrapped", "#C44E52")]  # seaborn "deep"


def load_rates() -> dict[tuple[str, str, str], float]:
    """Returns {(condition_label, group_label, base_model): misalign_rate}."""
    rates: dict[tuple[str, str, str], float] = {}
    for condition_label, _ in CONDITIONS:
        csv_path = results_root / condition_label / "em_ip_rfa.csv"
        df = pd.read_csv(csv_path)
        for base_model, _ in MODEL_LABELS:
            for group_label, group_prefix in GROUP_PREFIXES:
                mask = (
                    (df["base_model"] == base_model)
                    & (df["group"].str.startswith(group_prefix))
                )
                rate = df.loc[mask, "score"].mean()  # NaN excluded automatically
                rates[(condition_label, group_label, base_model)] = float(rate)
    return rates


def plot(rates: dict[tuple[str, str, str], float]) -> None:
    n_cols = 3
    n_rows = (len(GROUP_PREFIXES) + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 4.5 * n_rows), sharey=True
    )
    axes = axes_grid.flatten()

    x = np.arange(len(MODEL_LABELS))
    bar_width = 0.38

    for ax, (group_label, _) in zip(axes, GROUP_PREFIXES):
        for i, (condition_label, color) in enumerate(CONDITIONS):
            offset = (i - 0.5) * bar_width
            heights = [
                rates[(condition_label, group_label, base_model)]
                for base_model, _ in MODEL_LABELS
            ]
            bars = ax.bar(
                x + offset, heights, bar_width,
                label=condition_label, color=color,
            )
            for bar, h in zip(bars, heights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_title(group_label)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in MODEL_LABELS], rotation=15)
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False)

    # Hide any unused axes when GROUP_PREFIXES doesn't perfectly fill the grid.
    for ax in axes[len(GROUP_PREFIXES):]:
        ax.set_visible(False)

    for row in range(n_rows):
        axes_grid[row, 0].set_ylabel("P(misaligned)")

    fig.suptitle(
        "Free-form EM rate, risky_financial_advice: training conditions under clean vs wrapped",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    plot(load_rates())
