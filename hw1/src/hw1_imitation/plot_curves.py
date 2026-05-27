"""Generate training curves from an existing experiment ``log.csv``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from hw1_imitation.plotting import save_training_curves_from_csv


@dataclass
class PlotConfig:
    """Plot train/loss and eval/mean_reward vs step from a finished run."""

    # Path to ``exp/<run_name>/log.csv`` or the run directory containing it.
    log_path: Path
    # Output image path (default: same directory as log.csv).
    output: Path | None = None
    title: str | None = "Push-T MSE policy — training curves"


def main() -> None:
    config = tyro.cli(PlotConfig)
    log_path = config.log_path
    if log_path.is_dir():
        log_path = log_path / "log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"No log.csv at {log_path}")

    out = save_training_curves_from_csv(
        log_path,
        config.output,
        title=config.title,
    )
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
