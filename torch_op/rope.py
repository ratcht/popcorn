from typing import Annotated

import einops
import torch as t
import torch.nn as nn
from torch import Tensor

# dim = embedding_dim


class RotaryPositionEmbedding(nn.Module):
  cos_cached: t.Tensor
  sin_cached: t.Tensor

  def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
    super().__init__()
    self.dim = dim
    self.base = base

    # theta
    theta = self.base ** (-2 * t.arange(0, self.dim//2).float() / self.dim)
    assert theta.shape == (self.dim // 2,)

    # position indices
    m = t.arange(max_seq_len).float()
    assert m.shape == (max_seq_len,)

    # m * theta
    emb = einops.repeat(t.outer(m, theta), "s d -> s (d 2)")

    self.register_buffer("cos_cached", emb.cos())
    self.register_buffer("sin_cached", emb.sin())

  def forward(self, x: Annotated[Tensor, "batch seq_len d_model"]):
    B, L, D = x.shape
    assert D == self.dim

    cos = self.cos_cached[:L, :]
    sin = self.sin_cached[:L, :]

    assert cos.shape == sin.shape == (L, D)

    return self.apply_rotary_emb(x, cos, sin)

  def apply_rotary_emb(self, x, cos, sin):
    x_1, x_2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    x_r = t.cat((-x_2, x_1), dim=-1)

    return (x * cos) + (x_r * sin)


if __name__ == "__main__":
  from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

  rope = RotaryPositionEmbedding(4, 12)
  x = t.arange(0, 24, dtype=t.float32).reshape(1, 6, 4) # B L D

  output = rope(x)

  cos = rope.cos_cached[:6, :]
  sin = rope.sin_cached[:6, :]
  hf_output, _ = apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=0)

  # print("INPUT =========")
  # print(x)
  # print("OUTPUT ========")
  # print(output)
  # print("HF OUTPUT =====")
  # print(hf_output)

  assert t.allclose(output, hf_output)
  print("Successful!")
