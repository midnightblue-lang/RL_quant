from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only before installing torch
    torch = None


def discounted_returns(rewards: list[float], gamma: float = 1.0) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def ppo_losses(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for PPO losses.")
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns)
    entropy_bonus = entropy.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_bonus.detach().cpu()),
    }
    return loss, metrics
