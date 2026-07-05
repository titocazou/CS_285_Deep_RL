import torch
from torch import nn
import numpy as np

from infrastructure import pytorch_util as ptu


class DQNCritic(nn.Module):
    """Critic network for DQN. Maps observations to Q-values for each action."""

    def __init__(self, observation_shape, num_actions, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=int(np.prod(observation_shape)),
            output_size=num_actions,
            n_layers=n_layers,
            size=size,
        )

    def forward(self, obs):
        """
        Return Q-values for all actions.

        Args:
            obs: (batch_size, *observation_shape) observations
        Returns:
            qa_values: (batch_size, num_actions) Q-values for each action
        """
        # Flatten observations if needed
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.net(obs)


class DuelingDQNCritic(nn.Module):
    """Dueling critic for DQN.

    Shares a torso, then splits into a state-value head V(s) and a per-action
    advantage head A(s, a). The two are recombined into Q-values with a
    mean-centered advantage,

        Q(s, a) = V(s) + ( A(s, a) - mean_a' A(s, a') ),

    so that V and A are identifiable (only their sum is observed as Q, and the
    mean-centering pins down the split).
    """

    def __init__(self, observation_shape, num_actions, n_layers, size):
        super().__init__()
        input_size = int(np.prod(observation_shape))
        # Shared torso: same hidden stack as DQNCritic, but stopping at the last
        # hidden activation (size features) instead of projecting to num_actions.
        self.torso = ptu.build_mlp(
            input_size=input_size,
            output_size=size,
            n_layers=max(n_layers - 1, 0),
            size=size,
            output_activation="tanh",
        )
        self.value_head = nn.Linear(size, 1)
        self.advantage_head = nn.Linear(size, num_actions)

    def forward(self, obs):
        """
        Return Q-values for all actions.

        Args:
            obs: (batch_size, *observation_shape) observations
        Returns:
            qa_values: (batch_size, num_actions) Q-values for each action
        """
        # Flatten observations if needed
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        features = self.torso(obs)
        value = self.value_head(features)  # (batch_size, 1)
        advantage = self.advantage_head(features)  # (batch_size, num_actions)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class StateActionCritic(nn.Module):
    """Critic network for SAC. Maps (state, action) pairs to Q-values."""

    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )

    def forward(self, obs, acs):
        """
        Return Q-value for the given state-action pair.

        Args:
            obs: (batch_size, ob_dim) observations
            acs: (batch_size, ac_dim) actions
        Returns:
            q_values: (batch_size,) Q-values
        """
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)
