#!/usr/bin/env python3
"""Plot the evidence-dependence estimator fidelity bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


SIGNALS = [
    "Entropy",
    "Answer\nlogprob",
    "Maximum\nprob.",
    "Logit\nreadout",
    "Search-G1\nrepr.",
]

AUC = {
    "nq": [0.654, 0.572, 0.649, 0.638, 0.959],
    "2wiki": [0.565, 0.613, 0.552, 0.693, 0.871],
}

EDGE_COLORS = ["#8FA0AF", "#9AA5B1", "#8793A1", "#2F6DB5", "#0B9B62"]
FACE_COLORS = ["#F5F8FA", "#FAFBFC", "#F6F8FA", "#F3F8FF", "#E9FFF2"]
HATCHES = ["...", "//", "xxx", "\\\\", "///"]


def setup_style(font_path: Path) -> None:
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_family = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        font_family = "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 18,
            "axes.linewidth": 2.6,
            "axes.edgecolor": "#2B4059",
            "xtick.color": "#22344D",
            "ytick.color": "#22344D",
            "text.color": "#172338",
            "axes.labelcolor": "#172338",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_bars(ax: plt.Axes, values: list[float]) -> None:
    x = np.arange(len(values))
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=FACE_COLORS,
        edgecolor=EDGE_COLORS,
        linewidth=3.0,
    )
    for bar, hatch, edge_color in zip(bars, HATCHES, EDGE_COLORS):
        bar.set_hatch(hatch)
        bar.set_edgecolor(edge_color)

    ax.set_ylim(0.5, 1.02)
    yticks = np.arange(0.5, 1.01, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{value:.1f}" for value in yticks], fontsize=21, fontweight="bold")
    ax.grid(axis="y", linestyle=(0, (5, 5)), color="#DCE6EF", linewidth=1.6)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(SIGNALS, fontsize=18, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=7, width=2.2, pad=8)
    ax.set_ylabel("AUC", fontsize=27, fontweight="bold", labelpad=16)

    for xi, yi in zip(x, values):
        ax.text(
            xi,
            yi + 0.016,
            f"{yi:.3f}",
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.6)
    ax.spines["bottom"].set_linewidth(2.6)


def save_single(output_dir: Path, key: str, values: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.35))
    draw_bars(ax, values)
    fig.tight_layout(pad=0.9)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"estimator_auc_{key}{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_pair(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(21.0, 4.35))
    for ax, values in zip(axes, AUC.values()):
        draw_bars(ax, values)
    fig.tight_layout(w_pad=3.0, pad=0.9)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"estimator_auc_pair{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("assets/fonts/ComicNeue-Regular.ttf"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style(args.font)
    for key, values in AUC.items():
        save_single(args.out_dir, key, values)
    save_pair(args.out_dir)

    for path in sorted(args.out_dir.glob("estimator_auc_*")):
        print(path)


if __name__ == "__main__":
    main()
