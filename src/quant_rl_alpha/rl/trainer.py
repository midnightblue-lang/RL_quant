from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quant_rl_alpha.alpha import AlphaPool
from quant_rl_alpha.expression.tokens import BEG, ExpressionTokens, expression_tokens
from quant_rl_alpha.rl.env import AlphaMiningEnv
from quant_rl_alpha.rl.policy import PolicyValueNet, mask_logits
from quant_rl_alpha.rl.ppo import discounted_returns, ppo_losses
from quant_rl_alpha.utils.config import load_config

try:
    import torch
    from torch.distributions import Categorical
    from torch.nn.utils.rnn import pad_sequence
except ModuleNotFoundError:  # pragma: no cover - exercised only before installing torch
    torch = None
    Categorical = None
    pad_sequence = None


@dataclass
class EpisodeBatch:
    states: list[tuple[str, ...]]
    valid_masks: list[tuple[bool, ...]]
    actions: list[str]
    rewards: list[float]
    old_log_probs: list[float]
    terminal_infos: list[dict[str, object]]


class PPOTrainer:
    def __init__(
        self,
        env: AlphaMiningEnv,
        *,
        registry: ExpressionTokens | None = None,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for PPOTrainer.")
        config = config or load_config("rl")
        self.env = env
        self.registry = registry or expression_tokens()
        self.tokens = self.registry.all_tokens
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.device = torch.device(device or config["device"])
        self.gamma = float(config["gamma"])
        self.clip_epsilon = float(config["clip_epsilon"])
        self.ppo_epochs = int(config["ppo_epochs"])
        self.value_coef = float(config["value_coef"])
        self.entropy_coef = float(config["entropy_coef"])
        self.model = PolicyValueNet(
            len(self.tokens),
            lstm_layers=int(config["lstm_layers"]),
            lstm_hidden_size=int(config["lstm_hidden_size"]),
            dropout=float(config["dropout"]),
            head_hidden_size=int(config["head_hidden_size"]),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(config["learning_rate"])
        )

    def collect_episode(self) -> EpisodeBatch:
        state = self.env.reset()
        states: list[tuple[str, ...]] = []
        valid_masks: list[tuple[bool, ...]] = []
        actions: list[str] = []
        rewards: list[float] = []
        old_log_probs: list[float] = []
        terminal_infos: list[dict[str, object]] = []
        done = False
        while not done:
            valid_mask = self._valid_mask()
            action, log_prob = self._sample_action(state, valid_mask)
            next_state, reward, done, info = self.env.step(action)
            states.append(state)
            valid_masks.append(tuple(bool(item) for item in valid_mask.cpu().tolist()))
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_prob)
            if done:
                terminal_info = dict(info)
                terminal_info["reward"] = reward
                terminal_info["steps"] = len(states)
                terminal_infos.append(terminal_info)
            state = next_state
        return EpisodeBatch(states, valid_masks, actions, rewards, old_log_probs, terminal_infos)

    def update(self, batch: EpisodeBatch) -> dict[str, float]:
        states, lengths = self._state_tensor(batch.states)
        valid_masks = torch.tensor(batch.valid_masks, dtype=torch.bool, device=self.device)
        actions = torch.tensor(
            [self.token_to_id[token] for token in batch.actions], device=self.device
        )
        old_log_probs = torch.tensor(batch.old_log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(
            discounted_returns(batch.rewards, self.gamma), dtype=torch.float32, device=self.device
        )

        last_metrics: dict[str, float] = {}
        for _ in range(self.ppo_epochs):
            logits, values = self.model(states, lengths)
            dist = self._masked_distribution(logits, valid_masks)
            new_log_probs = dist.log_prob(actions)
            advantages = returns - values.detach()
            loss, last_metrics = ppo_losses(
                new_log_probs,
                old_log_probs,
                advantages,
                values,
                returns,
                dist.entropy(),
                clip_epsilon=self.clip_epsilon,
                value_coef=self.value_coef,
                entropy_coef=self.entropy_coef,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return last_metrics

    def train_iterations(
        self,
        iterations: int,
        episodes_per_iteration: int = 1,
        on_iteration: Callable[[int, dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        history = []
        for iteration in range(1, iterations + 1):
            batches = [self.collect_episode() for _ in range(episodes_per_iteration)]
            batch = _merge_batches(batches)
            metrics = self.update(batch)
            metrics.update(_episode_metrics(batch, self.env.pool))
            history.append(metrics)
            if on_iteration is not None:
                on_iteration(iteration, metrics)
        return history

    def _sample_action(
        self,
        state: tuple[str, ...],
        valid_mask: torch.Tensor,
    ) -> tuple[str, float]:
        states, lengths = self._state_tensor([state])
        logits, _ = self.model(states, lengths)
        dist = self._masked_distribution(logits[0], valid_mask)
        action_id = int(dist.sample().item())
        action_tensor = torch.tensor(action_id, device=self.device)
        log_prob = dist.log_prob(action_tensor).detach().cpu().item()
        return self.id_to_token[action_id], float(log_prob)

    def _state_tensor(self, states: list[tuple[str, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [
            torch.tensor(
                [self.token_to_id.get(token, self.token_to_id[BEG]) for token in state],
                device=self.device,
            )
            for state in states
        ]
        lengths = torch.tensor(
            [len(state) for state in states], dtype=torch.long, device=self.device
        )
        padded = pad_sequence(encoded, batch_first=True, padding_value=self.token_to_id[BEG])
        return padded, lengths

    def _valid_mask(self) -> torch.Tensor:
        valid_mask = torch.zeros(len(self.tokens), dtype=torch.bool, device=self.device)
        for token in self.env.valid_actions():
            valid_mask[self.token_to_id[token]] = True
        return valid_mask

    def _masked_distribution(self, logits: torch.Tensor, valid_mask: torch.Tensor) -> Categorical:
        return Categorical(logits=mask_logits(logits, valid_mask))


def _merge_batches(batches: list[EpisodeBatch]) -> EpisodeBatch:
    return EpisodeBatch(
        states=[state for batch in batches for state in batch.states],
        valid_masks=[mask for batch in batches for mask in batch.valid_masks],
        actions=[action for batch in batches for action in batch.actions],
        rewards=[reward for batch in batches for reward in batch.rewards],
        old_log_probs=[log_prob for batch in batches for log_prob in batch.old_log_probs],
        terminal_infos=[info for batch in batches for info in batch.terminal_infos],
    )


def _episode_metrics(batch: EpisodeBatch, pool: AlphaPool) -> dict[str, float]:
    infos = batch.terminal_infos
    rewards = [float(info["reward"]) for info in infos]
    valid_infos = [info for info in infos if bool(info.get("valid"))]
    invalid_infos = [info for info in infos if not bool(info.get("valid"))]
    pool_size = len(pool.entries)
    metrics = {
        "episodes": float(len(infos)),
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else float("nan"),
        "valid_ratio": len(valid_infos) / len(infos) if infos else float("nan"),
        "pool_size": float(pool_size),
        "invalid_count": float(len(invalid_infos)),
    }
    if valid_infos:
        metrics["pool_ic"] = float(valid_infos[-1]["pool_ic"])
        metrics["pool_loss"] = float(valid_infos[-1]["pool_loss"])
    else:
        metrics["pool_ic"] = float(pool.pool_ic()) if pool_size else 0.0
        metrics["pool_loss"] = float(pool.pool_loss()) if pool_size else 1.0
    return metrics
