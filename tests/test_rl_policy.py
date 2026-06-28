import importlib.util

import pytest

if importlib.util.find_spec("torch") is not None:
    import torch
    from torch.distributions import Categorical
else:
    torch = None
    Categorical = None

from quant_rl_alpha.rl.policy import PolicyValueNet, mask_logits


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_mask_logits_zeroes_invalid_batch_probabilities() -> None:
    logits = torch.zeros((2, 3))
    valid_mask = torch.tensor([[True, False, True], [False, True, False]])

    probs = Categorical(logits=mask_logits(logits, valid_mask)).probs

    assert probs[0, 1] == 0
    assert probs[1, 0] == 0
    assert probs[1, 2] == 0


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_policy_value_net_uses_lengths_for_padded_sequences() -> None:
    torch.manual_seed(7)
    model = PolicyValueNet(
        4,
        lstm_layers=1,
        lstm_hidden_size=8,
        dropout=0.0,
        head_hidden_size=4,
    )
    model.eval()
    padded = torch.tensor([[0, 1, 0], [0, 1, 2]])
    lengths = torch.tensor([2, 3])

    batch_logits, batch_values = model(padded, lengths)
    single_logits, single_values = model(torch.tensor([[0, 1]]), torch.tensor([2]))

    assert torch.allclose(batch_logits[0], single_logits[0], atol=1e-6)
    assert torch.allclose(batch_values[0], single_values[0], atol=1e-6)
