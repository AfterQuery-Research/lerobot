from __future__ import annotations

import pytest
import torch

from lerobot.policies.molmoact2.processor_molmoact2 import (
    MolmoAct2PackInputsProcessorStep,
    _pop_action_horizon_mask,
)


def test_action_padding_propagates_dataset_horizon_mask() -> None:
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.max_action_dim = 32
    action = torch.ones(2, 4, 7)
    action_is_pad = torch.tensor(
        [
            [False, False, True, True],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )

    padded, horizon_is_pad, dim_is_pad = step._pad_action(action, action_is_pad)

    assert padded.shape == (2, 4, 32)
    assert torch.equal(horizon_is_pad, action_is_pad)
    assert not dim_is_pad[:, :7].any()
    assert dim_is_pad[:, 7:].all()


def test_action_padding_preserves_legacy_unbatched_horizon_mask() -> None:
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.max_action_dim = 32
    action = torch.ones(4, 7)
    action_is_pad = torch.tensor([False, False, True, True], dtype=torch.bool)

    padded, horizon_is_pad, _ = step._pad_action(action, action_is_pad)

    assert padded.shape == (1, 4, 32)
    assert torch.equal(horizon_is_pad, action_is_pad.unsqueeze(0))


@pytest.mark.parametrize(
    "action_is_pad",
    [
        torch.zeros(2, 4, dtype=torch.int64),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool),
    ],
)
def test_action_padding_rejects_non_boolean_or_non_bxt_masks(action_is_pad: torch.Tensor) -> None:
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.max_action_dim = 32
    action = torch.ones(2, 4, 7)

    with pytest.raises(ValueError, match="action_is_pad"):
        step._pad_action(action, action_is_pad)


def test_action_padding_defaults_to_no_horizon_padding_for_generic_datasets() -> None:
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.max_action_dim = 32

    _, horizon_is_pad, _ = step._pad_action(torch.ones(2, 4, 7))

    assert horizon_is_pad.shape == (2, 4)
    assert not horizon_is_pad.any()


def test_processor_accepts_legacy_horizon_mask_alias() -> None:
    expected = torch.tensor([[False, True]], dtype=torch.bool)
    complementary = {"action_horizon_is_pad": expected.clone()}

    actual = _pop_action_horizon_mask(complementary)

    assert torch.equal(actual, expected)
    assert "action_horizon_is_pad" not in complementary


def test_action_padding_rejects_conflicting_dataset_and_legacy_masks() -> None:
    dataset_mask = torch.tensor([[False, True]], dtype=torch.bool)
    legacy_mask = torch.tensor([[False, False]], dtype=torch.bool)

    with pytest.raises(ValueError, match="disagree"):
        _pop_action_horizon_mask(
            {
                "action_is_pad": dataset_mask,
                "action_horizon_is_pad": legacy_mask,
            }
        )


def test_processor_accepts_identical_dataset_and_legacy_masks() -> None:
    expected = torch.tensor([[False, True]], dtype=torch.bool)
    complementary = {
        "action_is_pad": expected.clone(),
        "action_horizon_is_pad": expected.clone(),
    }

    actual = _pop_action_horizon_mask(complementary)

    assert torch.equal(actual, expected)
    assert complementary == {}
