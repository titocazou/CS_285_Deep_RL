from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        use_distributional: bool = False,
        use_noisy: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q
        self.use_distributional = use_distributional
        self.use_noisy = use_noisy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # NoisyNet: exploration comes from fresh weight noise, so resample before
        # acting. In eval mode the noise is off and the layers use their means.
        if self.use_noisy and self.training:
            self.critic.reset_noise()

        with torch.no_grad():
            q_values = self.critic.forward(observation)
        max_action = torch.argmax(q_values, dim=1).item()

        if np.random.rand() < 1 - epsilon:
             return max_action
        else:
            rand_actions = np.arange(self.num_actions)
            rand_actions = rand_actions[rand_actions != max_action]

            return np.random.choice(rand_actions)

    

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """Update the DQN critic, and return stats for logging.

        weights are the per-sample importance-sampling weights from prioritized
        replay (None for uniform replay). When prioritized, the returned dict
        also carries "td_errors", the per-sample priorities for the buffer.
        """
        if self.use_distributional:
            return self.update_critic_distributional(
                obs, action, reward, next_obs, done, weights
            )

        (batch_size,) = reward.shape
        batch_idx = torch.arange(batch_size, device=obs.device)

        # NoisyNet: draw one fresh set of perturbations per update, for both the
        # online and target critics.
        if self.use_noisy:
            self.critic.reset_noise()
            self.target_critic.reset_noise()

        # Compute target values
        with torch.no_grad():
            next_target_qa_values = self.target_critic.forward(next_obs)

            if self.use_double_q: 
                next_critic_qa_values = self.critic.forward(next_obs)
                next_action = torch.argmax(next_critic_qa_values, dim=1)
            else:
                next_action = torch.argmax(next_target_qa_values, dim=1)

           
            next_q_values = next_target_qa_values[batch_idx, next_action.long()]
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward + self.discount * (1 - done.float()) * next_q_values;
            assert target_values.shape == (batch_size,), target_values.shape

        qa_values = self.critic.forward(obs)
        q_values = qa_values[batch_idx, action.long()]

        td_error = q_values - target_values
        elementwise_loss = td_error ** 2
        if weights is not None:
            elementwise_loss = elementwise_loss * weights
        loss = elementwise_loss.mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        info = {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }
        if weights is not None:
            info["td_errors"] = td_error.detach().abs().cpu().numpy()
        return info

    def update_critic_distributional(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """C51 critic update.

        Fits the categorical return distribution to the projected distributional
        Bellman target, with a cross-entropy loss instead of the scalar MSE. The
        action selection is still greedy in the expected Q-values, so double-Q
        works exactly as before.
        """
        (batch_size,) = reward.shape
        batch_idx = torch.arange(batch_size, device=obs.device)

        support = self.critic.support  # (num_atoms,)
        num_atoms = self.critic.num_atoms
        v_min, v_max = self.critic.v_min, self.critic.v_max
        delta_z = (v_max - v_min) / (num_atoms - 1)

        with torch.no_grad():
            # Greedy next action from the expected Q-values. Under double-Q the
            # online net picks the action; the target net always supplies the
            # distribution.
            if self.use_double_q:
                next_action = torch.argmax(self.critic.forward(next_obs), dim=1)
            else:
                next_action = torch.argmax(self.target_critic.forward(next_obs), dim=1)

            next_probs = self.target_critic.dist(next_obs)  # (B, A, num_atoms)
            next_probs = next_probs[batch_idx, next_action.long()]  # (B, num_atoms)

            # Shift every atom by the Bellman update, then clamp back onto the
            # support. reward/done broadcast over the atom axis.
            reward = reward.unsqueeze(1)  # (B, 1)
            not_done = (1 - done.float()).unsqueeze(1)  # (B, 1)
            Tz = reward + self.discount * not_done * support.unsqueeze(0)  # (B, atoms)
            Tz = Tz.clamp(v_min, v_max)

            # Categorical projection: spread each shifted atom's probability onto
            # the two nearest fixed atoms.
            # b should land in [0, num_atoms - 1], but float32 rounding at the
            # support edges can push it a hair past the top atom (e.g. 50.0000038
            # for a 51-atom support), which makes ceil(b) == num_atoms and sends
            # the scatter index out of bounds. Clamp b to the valid range.
            b = (Tz - v_min) / delta_z  # (B, atoms), in [0, num_atoms - 1]
            b = b.clamp(0, num_atoms - 1)
            lower = b.floor().long()
            upper = b.ceil().long()

            # When b lands exactly on an atom, lower == upper and both weights are
            # zero; nudge the pair apart so that atom keeps its full mass.
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < num_atoms - 1) & (lower == upper)] += 1

            target_dist = torch.zeros_like(next_probs)  # (B, num_atoms)
            target_dist.scatter_add_(1, lower, next_probs * (upper.float() - b))
            target_dist.scatter_add_(1, upper, next_probs * (b - lower.float()))

        log_probs = torch.log(self.critic.dist(obs) + 1e-8)  # (B, A, num_atoms)
        log_probs = log_probs[batch_idx, action.long()]  # (B, num_atoms)
        cross_entropy = -(target_dist * log_probs).sum(dim=1)  # (B,)
        if weights is not None:
            loss = (cross_entropy * weights).mean()
        else:
            loss = cross_entropy.mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()
        self.lr_scheduler.step()

        with torch.no_grad():
            pred_probs = self.critic.dist(obs)[batch_idx, action.long()]
            q_values = (pred_probs * support).sum(dim=1)
            target_values = (target_dist * support).sum(dim=1)

        info = {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }
        if weights is not None:
            # The per-sample cross-entropy is the natural priority here (the KL
            # to the target distribution), standing in for |TD error|.
            info["td_errors"] = cross_entropy.detach().abs().cpu().numpy()
        return info

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
        weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        critic_stats = self.update_critic(obs, action, reward, next_obs, done, weights)
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
