"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy
from hw1_imitation.plotting import save_training_curves

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    # AdamW weight decay (L2 shrinkage on weights); not part of mse_loss itself.
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # Dropout after each hidden ReLU in the MLP
    policy_dropout: float = 0.1
    # Std of Gaussian noise on **normalized** training states only (targets unchanged). 0 disables.
    train_state_noise_std: float = 0.00
    # If True, multiply noise by per-sample scale from dataset (high at episode start, 0 at end).
    train_state_noise_episode_decay: bool = True
    # The number of epochs to train for.
    num_epochs: int = 800
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None
    # Cosine LR schedule after each optimizer step. Set False to use constant lr.
    use_cosine_lr_scheduler: bool = True
    # ``T_max`` for CosineAnnealingLR (steps per half-cosine period). None = full train run.
    lr_scheduler_t_max: int | None = None
    # Minimum learning rate for cosine schedule.
    lr_scheduler_eta_min: float = 0.0
    # Save training_curves.png / .pdf in the experiment log dir after training.
    save_training_plots: bool = True


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    steps_per_epoch = len(loader)
    total_train_steps = steps_per_epoch * config.num_epochs
    scheduler_t_max = (
        config.lr_scheduler_t_max
        if config.lr_scheduler_t_max is not None
        else total_train_steps
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
        dropout=config.policy_dropout,
    ).to(device)

    model.nn = torch.compile(model.nn) 

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    print("Training loop")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler: CosineAnnealingLR | None = None
    if config.use_cosine_lr_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_t_max,
            eta_min=config.lr_scheduler_eta_min,
        )
    step = 0

    for epoch in range(config.num_epochs):
        for batch in loader:
            states, actions, noise_scale = batch
            states = states.to(device)
            actions = actions.to(device)
            noise_scale = noise_scale.to(device)
            if config.train_state_noise_std > 0.0:
                if config.train_state_noise_episode_decay:
                    scale = noise_scale
                else:
                    scale = torch.ones_like(noise_scale)
                states = states + (
                    config.train_state_noise_std
                    * scale
                    * torch.randn_like(states)
                )

            loss = model.compute_loss(states, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step() 

            if step % config.log_interval == 0:
                logger.log({"train/loss": loss.item()}, step=step)
                print(f"Step {step}, Loss: {loss.item()}, Epoch: {epoch}")
             

            if step % config.eval_interval == 0:
                evaluate_policy(
                    model,
                    normalizer,
                    device,
                    config.chunk_size,
                    config.video_size,
                    config.num_video_episodes,
                    config.flow_num_steps,
                    step,
                    logger,
                )

            step += 1

    if config.save_training_plots:
        plot_path = log_dir / "training_curves.png"
        save_training_curves(logger.rows, plot_path)
        print(f"Saved training curves to {plot_path}")

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
