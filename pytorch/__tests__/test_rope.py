import torch as t
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from pytorch.rope import RotaryPositionEmbedding


class TestRotaryPositionEmbedding:
  """Test custom RoPE implementation against HuggingFace's apply_rotary_pos_emb"""

  def test_basic_rope(self):
    """Test basic functionality matches HuggingFace"""
    rope = RotaryPositionEmbedding(4, 12)
    x = t.arange(0, 24, dtype=t.float32).reshape(1, 6, 4)  # B L D

    output = rope(x)

    cos = rope.cos_cached[:6, :]
    sin = rope.sin_cached[:6, :]
    hf_output, _ = apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=0)

    assert t.allclose(output, hf_output)

  def test_different_dimensions(self):
    """Test various embedding dimensions"""
    for dim in [8, 16, 32, 64]:
      rope = RotaryPositionEmbedding(dim, 128)
      x = t.randn(2, 10, dim)

      output = rope(x)

      cos = rope.cos_cached[:10, :]
      sin = rope.sin_cached[:10, :]
      hf_output, _ = apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=0)

      assert t.allclose(output, hf_output, atol=1e-6)

  def test_different_seq_lengths(self):
    """Test various sequence lengths"""
    dim = 16
    rope = RotaryPositionEmbedding(dim, 256)

    for seq_len in [1, 5, 32, 128]:
      x = t.randn(1, seq_len, dim)

      output = rope(x)

      cos = rope.cos_cached[:seq_len, :]
      sin = rope.sin_cached[:seq_len, :]
      hf_output, _ = apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=0)

      assert t.allclose(output, hf_output, atol=1e-6)

  def test_different_batch_sizes(self):
    """Test various batch sizes"""
    dim = 16
    seq_len = 10
    rope = RotaryPositionEmbedding(dim, 128)

    for batch_size in [1, 2, 4, 8]:
      x = t.randn(batch_size, seq_len, dim)

      output = rope(x)

      cos = rope.cos_cached[:seq_len, :]
      sin = rope.sin_cached[:seq_len, :]
      hf_output, _ = apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=0)

      assert t.allclose(output, hf_output, atol=1e-6)

  def test_different_base_values(self):
    """Test different base values for theta computation"""
    dim = 16
    seq_len = 10

    for base in [10000, 500000, 1000000]:
      rope = RotaryPositionEmbedding(dim, 128, base=base)
      x = t.randn(2, seq_len, dim)

      output = rope(x)

      cos = rope.cos_cached[:seq_len, :]
      sin = rope.sin_cached[:seq_len, :]
      hf_output, _ = apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=0)

      assert t.allclose(output, hf_output, atol=1e-6)
