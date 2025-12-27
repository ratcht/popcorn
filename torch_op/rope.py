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
    emb = einops.repeat(t.outer(m, theta), "s d -> s (2 d)")

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
  from transformers.models.llama.configuration_llama import LlamaConfig
  from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
  )

  dim = 4
  max_seq_len = 12
  base = 10000

  rope = RotaryPositionEmbedding(dim, max_seq_len, base)
  x = t.arange(0, 24, dtype=t.float32).reshape(1, 6, 4)  # B L D

  # Use HuggingFace's rotary embedding to generate sin/cos
  config = LlamaConfig(
    head_dim=dim,
    max_position_embeddings=max_seq_len,
    rope_theta=base,
  )
  hf_rope = LlamaRotaryEmbedding(config)
  position_ids = t.arange(6).unsqueeze(0)  # shape: (1, seq_len)
  hf_cos, hf_sin = hf_rope(x, position_ids)

  my_cos = rope.cos_cached[:6, :]
  my_sin = rope.sin_cached[:6, :]

  print("SIN/COS COMPARISON ====")
  print(f"My cos shape: {my_cos.shape}, HF cos shape: {hf_cos.shape}")
  print(f"Cos close: {t.allclose(my_cos, hf_cos.squeeze(0), atol=1e-5)}")
  print(f"Sin close: {t.allclose(my_sin, hf_sin.squeeze(0), atol=1e-5)}")

  output = rope(x)
  hf_output, _ = apply_rotary_pos_emb(x, x, hf_cos, hf_sin)

  print("INPUT =========")
  print(x)
  print("OUTPUT ========")
  print(output)
  print("HF OUTPUT =====")
  print(hf_output)

  assert t.allclose(my_cos, hf_cos.squeeze(0), atol=1e-5), "Cos mismatch!"
  assert t.allclose(my_sin, hf_sin.squeeze(0), atol=1e-5), "Sin mismatch!"
  assert t.allclose(output, hf_output, atol=1e-5), "Output mismatch!"
  print("Successful!")
