import torch as t
import torch.nn.functional as F

from pytorch.conv2d import Conv2d


class TestConv2d:
  """Test custom Conv2d implementation against PyTorch's F.conv2d"""

  def test_basic_conv2d(self):
    """Test basic functionality matches PyTorch"""
    layer = Conv2d(2, 3, (2, 2), stride=(2, 2))
    x = t.arange(0, 64, dtype=t.float32).reshape(2, 2, 4, 4)

    output = layer(x)
    torch_output = F.conv2d(x, layer.weight, stride=(2, 2))

    assert t.allclose(output, torch_output)

  def test_with_padding(self):
    """Test convolution with padding"""
    layer = Conv2d(2, 3, (3, 3), stride=(1, 1), padding=(1, 1))
    x = t.randn(1, 2, 5, 5)

    output = layer(x)
    torch_output = F.conv2d(x, layer.weight, stride=(1, 1), padding=(1, 1))

    assert t.allclose(output, torch_output)

  def test_different_strides(self):
    """Test various stride configurations"""
    configs = [((1, 1), (0, 0)), ((2, 1), (1, 0)), ((1, 2), (0, 1)), ((3, 2), (2, 1))]

    for stride, padding in configs:
      layer = Conv2d(2, 2, (3, 3), stride=stride, padding=padding)
      x = t.randn(1, 2, 8, 8)

      output = layer(x)
      torch_output = F.conv2d(x, layer.weight, stride=stride, padding=padding)

      assert t.allclose(output, torch_output, atol=1e-6)

  def test_kernel_size_variations(self):
    """Test different kernel sizes"""
    for kernel_size in [(1, 1), (2, 2), (4, 4), (3, 2)]:
      layer = Conv2d(2, 2, kernel_size)
      x = t.randn(1, 2, 10, 10)

      output = layer(x)
      torch_output = F.conv2d(x, layer.weight)

      assert t.allclose(output, torch_output, atol=1e-6)
