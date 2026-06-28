from quant_rl_alpha.expression.action_mask import valid_action_mask, valid_actions
from quant_rl_alpha.expression.tokens import expression_tokens


def test_expression_token_registry_matches_config() -> None:
    registry = expression_tokens()

    assert registry.features == ("open", "close", "high", "low", "volume", "vwap")
    assert registry.time_deltas == ("10d", "20d", "30d", "40d", "50d")
    assert registry.constants == (
        "-30",
        "-10",
        "-5",
        "-2",
        "-1",
        "-0.5",
        "-0.01",
        "0.01",
        "0.5",
        "1",
        "2",
        "5",
        "10",
        "30",
    )
    assert registry.max_tokens == 20


def test_initial_state_only_allows_beg() -> None:
    assert valid_actions([]) == ["BEG"]
    assert valid_action_mask([])["BEG"]


def test_beg_state_allows_operands_but_not_sep_or_operators() -> None:
    actions = valid_actions(["BEG"])

    assert "close" in actions
    assert "1" in actions
    assert "SEP" not in actions
    assert "Add" not in actions


def test_complete_non_constant_expression_allows_sep() -> None:
    actions = valid_actions(["BEG", "close"])

    assert "SEP" in actions
    assert "10d" in actions
    assert "Abs" in actions
    assert "Add" not in actions


def test_time_delta_must_be_consumed_by_time_series_operator() -> None:
    actions = valid_actions(["BEG", "close", "10d"])

    assert "Mean" in actions
    assert "Ref" in actions
    assert "close" not in actions
    assert "SEP" not in actions
    assert "Add" not in actions


def test_binary_time_series_operator_needs_two_expressions_before_window() -> None:
    unary_actions = valid_actions(["BEG", "close", "10d"])
    binary_actions = valid_actions(["BEG", "close", "vwap", "10d"])
    mixed_dimension_actions = valid_actions(["BEG", "close", "volume", "10d"])

    assert "Corr" not in unary_actions
    assert "Cov" not in unary_actions
    assert "Corr" in binary_actions
    assert "Cov" in binary_actions
    assert "Corr" in mixed_dimension_actions
    assert "Cov" not in mixed_dimension_actions


def test_pure_constant_expression_cannot_terminate_or_apply_constant_operator() -> None:
    single_constant = valid_actions(["BEG", "1"])
    two_constants = valid_actions(["BEG", "1", "2"])

    assert "SEP" not in single_constant
    assert "Log" not in single_constant
    assert "10d" not in single_constant
    assert "Add" not in two_constants


def test_binary_operator_can_use_one_constant_and_one_feature() -> None:
    actions = valid_actions(["BEG", "close", "1"])

    assert "Add" not in actions
    assert "Sub" not in actions
    assert "Mul" in actions
    assert "Div" in actions
    assert "Log" not in actions


def test_action_mask_rejects_dimensionally_invalid_binary_operators() -> None:
    mixed = valid_actions(["BEG", "vwap", "volume"])
    price_and_constant = valid_actions(["BEG", "vwap", "0.5"])
    ratio_and_constant = valid_actions(["BEG", "close", "vwap", "Div", "1"])

    assert "Add" not in mixed
    assert "Sub" not in mixed
    assert "Mul" not in mixed
    assert "Div" not in mixed
    assert "Less" not in price_and_constant
    assert "Greater" not in price_and_constant
    assert "Sub" in ratio_and_constant
    assert "Less" in ratio_and_constant


def test_invalid_state_has_no_valid_actions() -> None:
    assert valid_actions(["BEG", "close", "10d", "close"]) == []
    assert valid_actions(["close"]) == []
    assert valid_actions(["BEG", "close", "SEP"]) == []


def test_valid_action_mask_matches_valid_actions() -> None:
    tokens = ["BEG", "close", "1"]
    mask = valid_action_mask(tokens)

    assert mask["Mul"]
    assert not mask["SEP"]
    assert {token for token, is_valid in mask.items() if is_valid} == set(valid_actions(tokens))


def test_max_tokens_counts_the_generated_sequence_including_beg() -> None:
    assert valid_actions(["BEG", *["close"] * 19]) == []
