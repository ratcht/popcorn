from typing import Annotated

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
  def __init__(self, embed_dim, head_size):
    super().__init__()

    self.embed_dim = embed_dim
    self.head_size = head_size

    self.W_Q = nn.Linear(embed_dim, head_size, bias=False)
    self.W_K = nn.Linear(embed_dim, head_size, bias=False)
    self.W_V = nn.Linear(embed_dim, head_size, bias=False)

    self.register_buffer("mask", None)

  def forward(
      self,
      x: Annotated[t.Tensor, "batch seq_len embed_dim"]
    ) -> Annotated[t.Tensor, "batch seq_len head_size"]:
    batch, seq_len, embed_dim = x.shape

    Q = self.W_Q(x) # batch seq_len head_size
    K = self.W_K(x)
    V = self.W_V(x)

    att: Annotated[t.Tensor, "batch seq_len seq_len"] = Q @ K.transpose(-2, -1) # (B S H) @ (B H S) -> B S S
    att = att / (self.head_size ** 0.5)

    if self.mask is None or self.mask.size(0) != seq_len:
      mask = t.tril(t.ones(seq_len, seq_len))
      self.mask = mask.unsqueeze(0)

    att = att.masked_fill(
      self.mask[:, :seq_len, :seq_len] == 0, float('-inf')
    )

    att = F.softmax(att, dim=-1)

    out = att @ V # (B S S) @ (B S H) -> (B S H)

    assert out.shape == (batch, seq_len, self.head_size)
    return out



if __name__ == "__main__":
  t.manual_seed(0)

  batch = 2
  seq_len = 4
  embed_dim = 8
  head_size = 8

  x = t.randn(batch, seq_len, embed_dim)

  head = SelfAttentionHead(embed_dim, head_size)

  # --- mine ---
  out = head(x)

  # --- torch ref ---
  Q = head.W_Q(x)
  K = head.W_K(x)
  V = head.W_V(x)

  out_ref = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=t.triu(t.full((seq_len, seq_len), float('-inf')), diagonal=1)
  )

  # --- check ---
  print("max diff:", (out - out_ref).abs().max())
  assert t.allclose(out, out_ref, atol=1e-6)
  print("validation passed")
