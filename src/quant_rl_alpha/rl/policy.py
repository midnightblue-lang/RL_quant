from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only before installing torch
    torch = None
    nn = None

BasePolicyValueNet = nn.Module if nn is not None else object


class PolicyValueNet(BasePolicyValueNet):  # type: ignore[misc, valid-type]
    def __init__(
        self,
        vocab_size: int,
        *,
        lstm_layers: int = 2,
        lstm_hidden_size: int = 128,
        dropout: float = 0.1,
        head_hidden_size: int = 64,
    ) -> None:
        if nn is None:
            raise ModuleNotFoundError(
                "PyTorch is required for PolicyValueNet. Install torch before phase 5 training."
            )
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, lstm_hidden_size)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.policy_head = _head(lstm_hidden_size, head_hidden_size, vocab_size)
        self.value_head = _head(lstm_hidden_size, head_hidden_size, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(token_ids)
        output, _ = self.lstm(embedded)
        if lengths is None:
            last = output[:, -1, :]
        else:
            last_index = (lengths.to(output.device).long().clamp_min(1) - 1).view(-1, 1, 1)
            last_index = last_index.expand(-1, 1, output.size(-1))
            last = output.gather(1, last_index).squeeze(1)
        return self.policy_head(last), self.value_head(last).squeeze(-1)


def mask_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for action masking.")
    return logits.masked_fill(~valid_mask.bool(), torch.finfo(logits.dtype).min)


def _head(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_size),
    )
