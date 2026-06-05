import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        obs = ptu.from_numpy(obs)
        action_distribution = self.forward(obs)
        action = action_distribution.sample()

        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor)->distributions.Distribution:
        """
        This function defines the forward pass of the network.
        """
        if self.discrete:
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            action_distribution = distributions.Normal(mean, std)

        return action_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        action_distribution = self.forward(obs)
        if self.discrete:
            log_prob = action_distribution.log_prob(actions)
        else:
            #The log-probability of the full action vector is the sum of the log-probabilities of each action
            log_prob = action_distribution.log_prob(actions).sum(dim=-1)
        loss = torch.mean(-log_prob * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
            "Log Prob": log_prob.mean().item(),
            "Grad Norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        }
