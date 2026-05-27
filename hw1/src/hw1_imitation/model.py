"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F
import random


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


def _make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
) -> nn.Sequential:
    """Each entry in ``hidden_dims`` is one ``Linear -> ReLU -> Dropout`` block; then the action head."""
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one hidden width.")

    def make_drop() -> nn.Module:
        return nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    layers: list[nn.Module] = []
    for width in hidden_dims:
        layers.extend(
            (
                nn.Linear(in_dim, width),
                nn.ReLU(),
                make_drop(),
            )
        )
        in_dim = width
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)

class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.nn = _make_mlp(
            state_dim, action_dim * chunk_size, hidden_dims, dropout
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_chunk = action_chunk.reshape(state.shape[0], -1)
        return F.mse_loss(self.nn(state), action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.nn(state).reshape(state.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.nn = _make_mlp(
            state_dim + action_dim * chunk_size + 1,
            action_dim * chunk_size,
            hidden_dims,
            dropout,
        )

    def _flow_network(
        self, state: torch.Tensor, a_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        t = t.reshape(batch_size, 1).to(device=state.device, dtype=state.dtype)
        return self.nn(torch.cat([state, a_t, t], dim=-1))

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_chunk = action_chunk.reshape(state.shape[0], -1)
        t = torch.rand(
            state.shape[0], 1, device=action_chunk.device, dtype=action_chunk.dtype
        )
        action_gaussian = torch.randn_like(action_chunk)
        a_t = action_chunk * t + (1 - t) * action_gaussian
        target_velocity = action_chunk - action_gaussian
        return F.mse_loss(self._flow_network(state, a_t, t), target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        flat_chunk_dim = self.chunk_size * self.action_dim
        dt = 1.0 / num_steps
        a_t = torch.randn(
            batch_size, flat_chunk_dim, device=state.device, dtype=state.dtype
        )
        t = torch.zeros(batch_size, 1, device=state.device, dtype=state.dtype)

        for _ in range(num_steps):
            a_t = a_t + dt * self._flow_network(state, a_t, t)
            t = t + dt
        return a_t.reshape(batch_size, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128, 128),
    dropout: float = 0.1,
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
