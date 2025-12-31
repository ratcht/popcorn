# implementation of https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
from typing import Annotated

import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Conv1d(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    k = 1. / (self.in_channels * self.kernel_size)

    self.weight = nn.Parameter(
      (2*t.rand(self.out_channels, self.in_channels, self.kernel_size) - 1) * (k ** 0.5)
    )

  def forward(self, x: Annotated[Tensor, "batch in_channels length"]) -> Annotated[Tensor, "batch out_channels out_length"]:
    batch, in_channels, length = x.shape

    x_padded = F.pad(x, (self.padding, self.padding))
    x_stride = x_padded.stride()

    out_length = (length + 2*self.padding - self.kernel_size)//self.stride + 1

    x_s = x_padded.as_strided(
      size=(batch, in_channels, out_length, self.kernel_size),
      stride=(x_stride[0], x_stride[1], self.stride * x_stride[-1], x_stride[-1])
    )
    x_s = einops.rearrange(x_s, "batch c_in out k -> batch 1 c_in out k")

    w_s = einops.rearrange(self.weight, "c_out c_in k -> 1 c_out c_in 1 k")

    out = x_s * w_s
    assert out.shape == (batch, self.out_channels, self.in_channels, out_length, self.kernel_size)

    return einops.reduce(out, "b c_out c_in out k -> b c_out out", "sum")
