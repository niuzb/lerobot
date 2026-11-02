import pytest
import torch

from lerobot.policies.dreamzero.dreamdojo_data import encode_so101_actions


def test_encode_so101_actions_uses_four_step_baselines_and_reserved_channels():
    actions = torch.arange(13, dtype=torch.float32).view(1, 13, 1).repeat(2, 1, 6)
    encoded = encode_so101_actions(actions)

    assert encoded.shape == (2, 12, 384)
    torch.testing.assert_close(encoded[0, :, 169], torch.tensor([1, 2, 3, 4] * 3).float())
    assert torch.count_nonzero(encoded[..., :169]) == 0
    assert torch.count_nonzero(encoded[..., 175:]) == 0


@pytest.mark.parametrize("actions", [torch.zeros(12, 6), torch.zeros(13, 52), torch.zeros(13, 6, dtype=torch.int64)])
def test_encode_so101_actions_rejects_invalid_inputs(actions):
    with pytest.raises(ValueError):
        encode_so101_actions(actions)


def test_encode_so101_actions_rejects_nonfinite_values():
    actions = torch.zeros(13, 6)
    actions[0, 0] = float("nan")
    with pytest.raises(ValueError):
        encode_so101_actions(actions)
