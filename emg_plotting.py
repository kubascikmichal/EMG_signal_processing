"""Shared plotting style and helpers for the EMG notebooks."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Categorical slots, always assigned in this order (colour-blind validated set).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8c8b86"
GRID = "#e4e3df"
SEQUENTIAL_CMAP = "Blues"


def set_style() -> None:
    """Thin marks, recessive grid and axes, fixed categorical colour order."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
        }
    )


def label_colors(labels: Sequence) -> Dict:
    """Map each label to a fixed categorical slot (by sorted order, never by frequency)."""
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(sorted(labels))}


def shade_segments(ax, t: np.ndarray, labels: np.ndarray, colors: Dict, names: Optional[Dict] = None, alpha: float = 0.15):
    """Shade contiguous runs of each label behind a time series and return legend handles."""
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], change, [len(labels)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        lab = labels[a]
        if lab in colors:
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=colors[lab], alpha=alpha, lw=0)
    names = names or {}
    return [Patch(color=c, alpha=0.5, label=names.get(k, str(k))) for k, c in colors.items()]


def plot_channels(t: np.ndarray, data: np.ndarray, ch_names: Sequence[str], title: str, ylabel: str = "µV", color: str = PALETTE[0], height: float = 1.1):
    """Small multiples: one row per channel with a shared time axis."""
    n = data.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.5, height * n + 0.8)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, x, name in zip(axes, data, ch_names):
        ax.plot(t, x, color=color, lw=0.5)
        ax.set_ylabel(f"{name}\n{ylabel}", rotation=0, ha="right", va="center")
    axes[0].set_title(title, loc="left")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig, axes


def plot_annotations(ax, annotations: pd.DataFrame, labels: bool = True, max_labels: int = 40) -> None:
    """Draw annotation markers (e.g. from BrainAccess `eeg.annotate`) as thin vertical lines."""
    for i, row in enumerate(annotations.itertuples()):
        ax.axvline(row.onset_s, color=INK_MUTED, lw=0.8, ls="--")
        if labels and i < max_labels:
            ax.text(row.onset_s, 0.98, f"{row.label} ", transform=ax.get_xaxis_transform(), fontsize=7,
                    color=INK_SECONDARY, rotation=90, ha="right", va="top")


def plot_confusion(cm: np.ndarray, labels: Sequence[str], title: str):
    """Row-normalised confusion matrix with a single-hue sequential scale."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 2, 0.8 * len(labels) + 1.5))
    im = ax.imshow(cm_norm, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm_norm[i, j] > 0.6 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    return fig, ax
