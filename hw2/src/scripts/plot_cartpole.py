"""Plot CartPole learning curves from exp/cartpole and exp/cartpole_lb."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# CartPole-v0_cartpole_lb_rtg_na_sd1_20260520_231751
RUN_DIR_RE = re.compile(r"^CartPole-v0_(.+)_sd\d+_(\d{8}_\d{6})$")

HW2_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_STEPS_KEY = "Train_EnvstepsSoFar"
ITER_KEY = "step"
GRAD_KEY = "Grad Norm"


def parse_run_dir(dirname: str):
    match = RUN_DIR_RE.match(dirname)
    if not match:
        return None
    return match.group(1), match.group(2)


def load_curve(log_csv: Path, x_key: str, y_key: str):
    xs, ys = [], []
    with log_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            x = row.get(x_key)
            y = row.get(y_key)
            if x is None or y is None or x == "" or y == "":
                continue
            xs.append(float(x))
            ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def latest_run_dirs(exp_subdir: str) -> dict[str, Path]:
    """Most recent run folder per experiment type (e.g. cartpole_rtg_na)."""
    root = HW2_ROOT / "exp" / exp_subdir
    if not root.is_dir():
        return {}

    best: dict[str, tuple[str, Path]] = {}
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        parsed = parse_run_dir(run_dir.name)
        if parsed is None:
            continue
        exp_type, timestamp = parsed
        if exp_type not in best or timestamp > best[exp_type][0]:
            best[exp_type] = (timestamp, run_dir)

    return {exp_type: path for exp_type, (_, path) in best.items()}


def normalize_series(values: np.ndarray, method: str) -> np.ndarray:
    """Per-run normalization so curves are comparable on one axes."""
    if method == "none":
        return values
    if method == "zscore":
        std = float(values.std())
        if std < 1e-8:
            return values - values.mean()
        return (values - values.mean()) / std
    if method == "mean":
        mean = float(values.mean())
        if mean < 1e-8:
            return values
        return values / mean
    if method == "max":
        peak = float(values.max())
        if peak < 1e-8:
            return values
        return values / peak
    raise ValueError(f"Unknown normalize method: {method}")


def grad_norm_stats(values: np.ndarray) -> dict[str, float]:
    mean = float(values.mean())
    std = float(values.std())
    cv = std / mean if mean > 1e-8 else float("nan")
    return {"mean": mean, "std": std, "cv": cv}


def plot_group(
    exp_subdir: str,
    title: str,
    output: Path,
    x_key: str,
    x_label: str,
    y_key: str,
    y_label: str,
):
    runs = latest_run_dirs(exp_subdir)
    if not runs:
        print(f"No runs found under {HW2_ROOT / 'exp' / exp_subdir}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for exp_type in sorted(runs):
        log_csv = runs[exp_type] / "log.csv"
        if not log_csv.is_file():
            print(f"Skipping {exp_type}: no log.csv")
            continue
        xs, ys = load_curve(log_csv, x_key, y_key)
        if len(xs) == 0:
            print(f"Skipping {exp_type}: no {y_key} in log")
            continue
        ax.plot(xs, ys, label=exp_type)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"No curves plotted for {output.name}")
        return

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_grad_norm_group(
    exp_subdir: str,
    title: str,
    output: Path,
    normalize: str,
    stats_rows: list[dict],
):
    runs = latest_run_dirs(exp_subdir)
    if not runs:
        print(f"No runs found under {HW2_ROOT / 'exp' / exp_subdir}")
        return

    y_labels = {
        "none": "Grad Norm",
        "zscore": "Grad Norm (z-score per run)",
        "mean": "Grad Norm / mean",
        "max": "Grad Norm / max",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for exp_type in sorted(runs):
        log_csv = runs[exp_type] / "log.csv"
        if not log_csv.is_file():
            continue
        xs, ys = load_curve(log_csv, ITER_KEY, GRAD_KEY)
        if len(xs) == 0:
            continue

        raw_stats = grad_norm_stats(ys)
        stats_rows.append(
            {
                "group": exp_subdir,
                "exp_type": exp_type,
                "grad_norm_mean": raw_stats["mean"],
                "grad_norm_std": raw_stats["std"],
                "grad_norm_cv": raw_stats["cv"],
            }
        )

        ys_plot = normalize_series(ys, normalize)
        ax.plot(xs, ys_plot, label=exp_type)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"No grad norm curves plotted for {output.name}")
        return

    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_labels[normalize])
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def write_grad_norm_stats(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "exp_type",
        "grad_norm_mean",
        "grad_norm_std",
        "grad_norm_cv",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    print("\nGrad norm stats (raw, across iterations):")
    for row in rows:
        print(
            f"  [{row['group']}] {row['exp_type']}: "
            f"mean={row['grad_norm_mean']:.4f}, "
            f"std={row['grad_norm_std']:.4f}, "
            f"cv={row['grad_norm_cv']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HW2_ROOT / "exp" / "plots",
        help="Where to save PNGs",
    )
    parser.add_argument(
        "--grad-norm-normalize",
        choices=["zscore", "mean", "max", "none"],
        default="zscore",
        help=(
            "Per-run normalization for grad-norm plots. "
            "zscore (default) compares relative fluctuation; "
            "use none for raw amplitude (e.g. NA vs non-NA scale)."
        ),
    )
    parser.add_argument(
        "--grad-norm-raw-plot",
        action="store_true",
        help="Also save a second grad-norm figure with raw (unnormalized) values.",
    )
    args = parser.parse_args()
    out = args.out_dir

    plot_group(
        "cartpole",
        "CartPole (small batch) — Train average return",
        out / "cartpole_small_batch.png",
        ENV_STEPS_KEY,
        ENV_STEPS_KEY,
        "Train_AverageReturn",
        "Train_AverageReturn",
    )
    plot_group(
        "cartpole_lb",
        "CartPole (large batch) — Train average return",
        out / "cartpole_large_batch.png",
        ENV_STEPS_KEY,
        ENV_STEPS_KEY,
        "Train_AverageReturn",
        "Train_AverageReturn",
    )

    stats_rows: list[dict] = []
    norm = args.grad_norm_normalize
    plot_grad_norm_group(
        "cartpole",
        f"CartPole (small batch) — Gradient norm ({norm})",
        out / "cartpole_small_batch_grad_norm.png",
        norm,
        stats_rows,
    )
    plot_grad_norm_group(
        "cartpole_lb",
        f"CartPole (large batch) — Gradient norm ({norm})",
        out / "cartpole_large_batch_grad_norm.png",
        norm,
        stats_rows,
    )
    write_grad_norm_stats(out / "grad_norm_stats.csv", stats_rows)

    if args.grad_norm_raw_plot and norm != "none":
        raw_rows: list[dict] = []
        plot_grad_norm_group(
            "cartpole",
            "CartPole (small batch) — Gradient norm (raw)",
            out / "cartpole_small_batch_grad_norm_raw.png",
            "none",
            raw_rows,
        )
        plot_grad_norm_group(
            "cartpole_lb",
            "CartPole (large batch) — Gradient norm (raw)",
            out / "cartpole_large_batch_grad_norm_raw.png",
            "none",
            raw_rows,
        )


if __name__ == "__main__":
    main()
