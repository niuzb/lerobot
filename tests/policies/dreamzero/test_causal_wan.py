import torch
from torch import nn

from lerobot.policies.dreamzero.wan.causal_wan import CausalWanSelfAttention


class RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, query, key, value):
        self.calls.append((query.shape[1], key.shape[1], value.shape[1]))
        return torch.zeros_like(query)


def test_online_state_queries_only_see_the_current_state_block():
    attention = CausalWanSelfAttention(
        dim=8, num_heads=2, frame_seqlen=1, qk_norm=False, num_action_per_block=4, num_state_per_block=2
    )
    recorder = RecordingAttention()
    attention.attn = recorder
    image_query = torch.randn(1, 3, 2, 4)
    action_state_query = torch.randn(1, 6, 2, 4)
    image_key = torch.randn(1, 5, 2, 4)
    image_value = torch.randn_like(image_key)
    action_state_key = torch.randn_like(action_state_query)
    action_state_value = torch.randn_like(action_state_query)

    output = attention._online_action_state_attention(
        image_query, action_state_query, image_key, image_value, action_state_key, action_state_value
    )

    assert output.shape == (1, 9, 2, 4)
    assert recorder.calls == [(7, 11, 11), (2, 2, 2)]
