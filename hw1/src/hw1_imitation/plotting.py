"""Publication-style training curves (loss + eval reward vs step)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

TRAIN_LOSS_KEY = "train/loss"
EVAL_REWARD_KEY = "eval/mean_reward"

# Muted, print-friendly palette
COLOR_LOSS = "#2563eb"
COLOR_REWARD = "#059669"
COLOR_LOSS_RAW = "#93c5fd"
COLOR_GRID = "#e5e7eb"


def _coerce_float(value: Any) -> float | None:
    """Parse scalars from logger rows or CSV strings."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_series(
    rows: list[dict[str, Any]],
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for a scalar metric from logger rows or CSV rows."""
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        if metric_key not in row:
            continue
        step_val = _coerce_float(row.get("step"))
        metric_val = _coerce_float(row[metric_key])
        if step_val is None or metric_val is None:
            continue
        steps.append(int(step_val))
        values.append(metric_val)
    if not steps:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(values)[order]


def load_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _ema(values: np.ndarray, alpha: float = 0.92) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * values[i]
    return out


def save_training_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str | None = "Push-T MSE policy — training curves",
    loss_smooth_alpha: float = 0.92,
    dpi: int = 200,
) -> Path:
    """Plot training loss and eval mean reward vs step; save PNG (+ PDF sibling)."""
    train_steps, train_loss = extract_series(rows, TRAIN_LOSS_KEY)
    eval_steps, eval_reward = extract_series(rows, EVAL_REWARD_KEY)

    if len(train_steps) == 0 and len(eval_steps) == 0:
        raise ValueError(
            "No scalar metrics found to plot (train/loss or eval/mean_reward). "
            "Check that log.csv has numeric values and the expected column names."
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        sharex=True,
        constrained_layout=True,
    )

    # --- Training loss ---
    ax_loss = axes[0]
    if len(train_steps) > 0:
        ax_loss.plot(
            train_steps,
            train_loss,
            color=COLOR_LOSS_RAW,
            linewidth=0.9,
            alpha=0.45,
            label="Minibatch loss (logged)",
            zorder=1,
        )
        if len(train_steps) >= 3:
            smooth = _ema(train_loss, alpha=loss_smooth_alpha)
            ax_loss.plot(
                train_steps,
                smooth,
                color=COLOR_LOSS,
                linewidth=2.4,
                label="Smoothed (EMA)",
                zorder=2,
            )
        else:
            ax_loss.plot(
                train_steps,
                train_loss,
                color=COLOR_LOSS,
                linewidth=2.4,
                label="Training loss",
                zorder=2,
            )
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.95)

    # --- Eval reward ---
    ax_reward = axes[1]
    if len(eval_steps) > 0:
        ax_reward.plot(
            eval_steps,
            eval_reward,
            color=COLOR_REWARD,
            linewidth=2.2,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor=COLOR_REWARD,
            label="Eval mean max reward",
            zorder=2,
        )
        ax_reward.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.95)
    else:
        ax_reward.text(
            0.5,
            0.5,
            "No eval/mean_reward in log.csv\n"
            "(older runs only logged train loss to CSV;\n"
            "re-train with current code or check WandB)",
            transform=ax_reward.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#6b7280",
        )
    ax_reward.set_xlabel("Training step")
    ax_reward.set_ylabel("Mean max reward")
    ax_reward.set_title("Evaluation in simulator")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="600", y=1.02)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_training_curves_from_csv(
    csv_path: Path,
    output_path: Path | None = None,
    **kwargs: Any,
) -> Path:
    """Load ``log.csv`` from an experiment directory and write curves."""
    csv_path = Path(csv_path)
    rows = load_rows_from_csv(csv_path)
    if output_path is None:
        output_path = csv_path.parent / "training_curves.png"
    return save_training_curves(rows, output_path, **kwargs)
