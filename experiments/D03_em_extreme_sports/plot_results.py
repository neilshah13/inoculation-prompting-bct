"""Render two plots from the D03 cached judged JSONL files:

1. clean_vs_wrapped_em.png — bar chart, mean P(misaligned) per (base_model, group)
   under {clean, wrapped} system prompt. The headline re-elicit figure.
2. d01_vs_d03_em.png — comparison across datasets at clean (uninoculated baseline)
   to motivate using the Model-Organisms text datasets over insecure_code.

Outputs land in experiments/D03_em_extreme_sports/results/plots/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


D03_ROOT = Path(__file__).parent / "results" / "em_ip_es"
PLOT_DIR = Path(__file__).parent / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

URI_TO_LABEL = {
    "15f8d299ae887089": ("EM", "Llama-3.1-8B"),
    "cd68b4231ac25d62": ("IP", "Llama-3.1-8B"),
    "e2f85c11fd5c5018": ("CT+IP", "Llama-3.1-8B"),
    "b464931760b86c87": ("IP+control", "Llama-3.1-8B"),
    "57e8e01be0667288": ("EM", "Qwen3-8B"),
    "3b80b8fbce3caafc": ("IP", "Qwen3-8B"),
    "b987314c702d4f39": ("CT+IP", "Qwen3-8B"),
    "bbaa323df0155c40": ("IP+control", "Qwen3-8B"),
    "546f64b6a37c5f0d": ("EM", "Qwen3-32B"),
    "8eeb11e85484b2b7": ("IP", "Qwen3-32B"),
    "8e2b4e618fabb156": ("CT+IP", "Qwen3-32B"),
    "928edccaafbec062": ("IP+control", "Qwen3-32B"),
}


def label_for(fn: str):
    for k, v in URI_TO_LABEL.items():
        if k in fn:
            return v
    return None


def em_rate(p: Path) -> float | None:
    n_t = n_m = 0
    with open(p) as f:
        for line in f:
            for si in json.loads(line).get("score_infos", []):
                s = si.get("score")
                if s is None:
                    continue
                n_t += 1
                if bool(s):
                    n_m += 1
    return (n_m / n_t) if n_t else None


def load_rates(root: Path) -> dict:
    rates = {}
    for mode in ["clean", "wrapped", "control_wrapped"]:
        md = root / mode
        if not md.exists():
            continue
        em_dir = next(
            (d for d in md.iterdir() if d.is_dir() and d.name.startswith("emergent-misalignment")),
            None,
        )
        if em_dir is None:
            continue
        for cell in sorted(em_dir.iterdir()):
            if cell.name.endswith(".responses.jsonl") or not cell.name.endswith(".jsonl"):
                continue
            label = label_for(cell.name)
            if label is None:
                continue
            r = em_rate(cell)
            if r is None:
                continue
            rates[(mode, label[0], label[1])] = r
    return rates


def plot_clean_vs_wrapped(rates: dict, out: Path):
    models = ["Llama-3.1-8B", "Qwen3-8B", "Qwen3-32B"]
    # (panel_title, clean_mode, wrapped_mode) — IP+control panel uses control_wrapped
    panels = [
        ("EM",         "clean", "wrapped"),
        ("IP",         "clean", "wrapped"),
        ("IP+control", "clean", "control_wrapped"),
        ("CT+IP",      "clean", "wrapped"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    colors = {"clean": "#4a90d9", "wrapped": "#d96060"}

    for ax, (group, clean_mode, wrapped_mode) in zip(axes.flat, panels):
        x = np.arange(len(models))
        width = 0.36
        clean_vals = [rates.get((clean_mode, group, m), 0.0) for m in models]
        wrapped_vals = [rates.get((wrapped_mode, group, m), 0.0) for m in models]

        wrapped_label = "wrapped" if wrapped_mode == "wrapped" else "control-wrapped"
        b1 = ax.bar(x - width / 2, clean_vals, width, label="clean", color=colors["clean"])
        b2 = ax.bar(x + width / 2, wrapped_vals, width, label=wrapped_label, color=colors["wrapped"])

        for bar in (*b1, *b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylim(0, 0.65)
        ax.set_ylabel("P(misaligned)" if group == "EM" else "")
        ax.set_title(f"{group}")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Free-form EM rate, extreme_sports: EM, IP, IP+control, CT+IP under clean vs wrapped",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_re_elicit_gap(rates: dict, out: Path):
    models = ["Llama-3.1-8B", "Qwen3-8B", "Qwen3-32B"]
    em_gaps = [rates.get(("wrapped", "EM", m), 0) - rates.get(("clean", "EM", m), 0) for m in models]
    ip_gaps = [rates.get(("wrapped", "IP", m), 0) - rates.get(("clean", "IP", m), 0) for m in models]
    ct_gaps = [rates.get(("wrapped", "CT+IP", m), 0) - rates.get(("clean", "CT+IP", m), 0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(models))
    width = 0.27
    b1 = ax.bar(x - width, em_gaps, width, label="EM", color="#6c757d")
    b2 = ax.bar(x,         ip_gaps, width, label="IP", color="#d96060")
    b3 = ax.bar(x + width, ct_gaps, width, label="CT+IP", color="#3aa14a")
    for bar in (*b1, *b2, *b3):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, max(h, 0) + 0.008,
                f"+{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Re-elicit gap (wrapped − clean)")
    ax.set_title("Re-elicit gap on extreme_sports: CT+IP closes the gap IP leaves")
    ax.set_ylim(min(0, min(ct_gaps) - 0.02), max(em_gaps + ip_gaps + ct_gaps) * 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rates = load_rates(D03_ROOT)
    print("rates loaded:", len(rates), "cells")
    plot_clean_vs_wrapped(rates, PLOT_DIR / "clean_vs_wrapped_em.png")
    plot_re_elicit_gap(rates, PLOT_DIR / "re_elicit_gap.png")


if __name__ == "__main__":
    main()
