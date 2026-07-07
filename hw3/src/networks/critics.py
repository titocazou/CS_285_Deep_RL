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


class DistributionalDQNCritic(nn.Module):
    """Categorical (C51) critic for DQN.

    Instead of a single scalar Q(s, a), the network predicts a probability
    distribution over a fixed grid of return values (the "atoms"). The support
    is num_atoms points evenly spaced on [v_min, v_max]; for each action the
    head outputs num_atoms logits, softmaxed into probabilities p_i(s, a) over
    the atoms z_i. The scalar Q used for action selection is the mean of that
    distribution,

        Q(s, a) = sum_i z_i * p_i(s, a),

    so the greedy policy and the double-Q argmax are unchanged.
    """

    def __init__(
        self,
        observation_shape,
        num_actions,
        n_layers,
        size,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        # One head over all actions and atoms; reshaped to (A, num_atoms) below.
        self.net = ptu.build_mlp(
            input_size=int(np.prod(observation_shape)),
            output_size=num_actions * num_atoms,
            n_layers=n_layers,
            size=size,
        )
        # Fixed atom locations z_0 ... z_{num_atoms-1}, shared across states and
        # actions. Registered as a buffer so it moves with .to(device) and is
        # saved in the state dict.
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def dist(self, obs):
        """
        Return the per-action probability distribution over atoms.

        Args:
            obs: (batch_size, *observation_shape) observations
        Returns:
            probs: (batch_size, num_actions, num_atoms), summing to 1 over the
                last dimension.
        """
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        logits = self.net(obs).reshape(-1, self.num_actions, self.num_atoms)
        return torch.softmax(logits, dim=-1)

    def forward(self, obs):
        """
        Return the expected Q-values for all actions.

        Collapses each categorical distribution to its mean, so callers that
        only need a scalar Q (action selection, double-Q argmax) work unchanged.

        Args:
            obs: (batch_size, *observation_shape) observations
        Returns:
            qa_values: (batch_size, num_actions) Q-values for each action
        """
        probs = self.dist(obs)  # (batch_size, num_actions, num_atoms)
        return torch.sum(probs * self.support, dim=-1)


class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise on its weights (NoisyNet).

    Replaces the deterministic y = W x + b with

        y = (mu_W + sigma_W * eps_W) x + (mu_b + sigma_b * eps_b),

    where mu_* and sigma_* are learned and eps_* is resampled noise. The scale
    of the perturbation is therefore learned per weight: sigma can shrink toward
    zero where the network wants to be deterministic. Factorized noise keeps the
    sampling cheap: eps_W[i, j] = f(eps_in[j]) * f(eps_out[i]) with
    f(x) = sign(x) * sqrt(|x|), so we draw in + out unit-Gaussian numbers per
    step instead of in * out.

    In eval mode the noise is dropped and only the mean weights mu_* are used,
    so evaluation is deterministic.
    """

    def __init__(self, in_features, out_features, sigma_0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # Learned mean and noise-scale for the weights and biases.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise samples: not trained, resampled by reset_noise(). Registered as
        # buffers so they move with .to(device) and are saved in the state dict.
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # mu ~ U[-1/sqrt(p), 1/sqrt(p)] and sigma = sigma_0 / sqrt(p), with
        # p = in_features, following the factorized-noise setup in the paper.
        bound = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_0 * bound)
        self.bias_sigma.data.fill_(self.sigma_0 * bound)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Draw a fresh set of perturbations."""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        # Factorized outer product for the weight noise; the bias reuses eps_out.
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class NoisyDQNCritic(nn.Module):
    """DQN critic whose linear layers are NoisyLinear layers (NoisyNet).

    Same MLP shape as DQNCritic, but every Linear is replaced by a NoisyLinear,
    so exploration comes from learned weight noise instead of epsilon-greedy.
    Call reset_noise() to draw a fresh set of perturbations for the whole net.
    """

    def __init__(self, observation_shape, num_actions, n_layers, size, sigma_0=0.5):
        super().__init__()
        input_size = int(np.prod(observation_shape))
        layers = []
        in_size = input_size
        for _ in range(n_layers):
            layers.append(NoisyLinear(in_size, size, sigma_0=sigma_0))
            layers.append(nn.Tanh())
            in_size = size
        layers.append(NoisyLinear(in_size, num_actions, sigma_0=sigma_0))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        """
        Return Q-values for all actions.

        Args:
            obs: (batch_size, *observation_shape) observations
        Returns:
            qa_values: (batch_size, num_actions) Q-values for each action
        """
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.net(obs)

    def reset_noise(self):
        for module in self.net:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PreprocessAtari(nn.Module):
    """Scale uint8 Atari frames (0-255) to floats in [0, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4], f"Bad observation shape: {x.shape}"
        assert x.shape[-3:] == (4, 84, 84), f"Bad observation shape: {x.shape}"
        assert x.dtype == torch.uint8

        return x / 255.0


class AtariCritic(nn.Module):
    """Convolutional DQN critic for Atari, with optional dueling, distributional
    (C51) and noisy-net heads.

    The trunk is the standard 3-layer DQN encoder. The flags change only the
    fully-connected head, so any combination works, up to the full Rainbow head
    (dueling + distributional + noisy). With no flags set this is the plain
    Atari DQN critic. Non-distributional critics are the num_atoms == 1 special
    case of the same head, so one code path covers every combination.
    """

    def __init__(
        self,
        observation_shape,
        num_actions,
        use_dueling=False,
        use_distributional=False,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        use_noisy=False,
        sigma_0=0.5,
    ):
        super().__init__()
        assert observation_shape == (4, 84, 84), f"Observation shape: {observation_shape}"
        self.num_actions = num_actions
        self.use_dueling = use_dueling
        self.use_distributional = use_distributional
        self.use_noisy = use_noisy
        self.num_atoms = num_atoms if use_distributional else 1

        if use_distributional:
            self.v_min = v_min
            self.v_max = v_max
            self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        self.trunk = nn.Sequential(
            PreprocessAtari(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        feature_dim = 3136  # 64 * 7 * 7 for an 84x84 input

        # Swap in NoisyLinear for the head's linear layers when noisy is on.
        def fc(in_features, out_features):
            if use_noisy:
                return NoisyLinear(in_features, out_features, sigma_0=sigma_0)
            return nn.Linear(in_features, out_features)

        atoms = self.num_atoms
        if use_dueling:
            self.value_hidden = fc(feature_dim, 512)
            self.value_head = fc(512, atoms)  # V(s): (B, atoms)
            self.advantage_hidden = fc(feature_dim, 512)
            self.advantage_head = fc(512, num_actions * atoms)  # A(s, a)
        else:
            self.head_hidden = fc(feature_dim, 512)
            self.head = fc(512, num_actions * atoms)
        self.relu = nn.ReLU()

        self.to(ptu.device)

    def _logits(self, obs):
        """Head output as (batch_size, num_actions, num_atoms) logits."""
        features = self.trunk(obs)
        if self.use_dueling:
            value = self.value_head(self.relu(self.value_hidden(features)))
            advantage = self.advantage_head(self.relu(self.advantage_hidden(features)))
            value = value.view(-1, 1, self.num_atoms)
            advantage = advantage.view(-1, self.num_actions, self.num_atoms)
            # Mean-centered advantage, per atom, exactly as in the vector critic.
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        hidden = self.relu(self.head_hidden(features))
        return self.head(hidden).view(-1, self.num_actions, self.num_atoms)

    def dist(self, obs):
        """Per-action distribution over atoms, (batch_size, num_actions, num_atoms)."""
        return torch.softmax(self._logits(obs), dim=-1)

    def forward(self, obs):
        """Expected Q-values for all actions, (batch_size, num_actions)."""
        logits = self._logits(obs)
        if self.use_distributional:
            probs = torch.softmax(logits, dim=-1)
            return torch.sum(probs * self.support, dim=-1)
        # num_atoms == 1: the logits are the Q-values directly.
        return logits.squeeze(-1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


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
