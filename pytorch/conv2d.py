# implementation of https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
from typing import Annotated, Tuple

import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Conv2d(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: Annotated[Tuple, "height width"],
    stride: Annotated[Tuple, "height width"]=(1,1),
    padding: Annotated[Tuple, "height width"]=(0,0)
  ):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    k = 1. / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])

    self.weight = nn.Parameter(
      (2*t.rand(self.out_channels, self.in_channels, *self.kernel_size) - 1) * (k ** 0.5)
    )

  def forward(self, x: Annotated[Tensor, "batch in_channels height width"]) -> Annotated[Tensor, "batch out_channels out_height out_width"]:
    batch, in_channels, height, width = x.shape

    x_padded = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
    x_stride = x_padded.stride()

    out_height = (height + 2*self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
    out_width = (width + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1

    print(x_stride)
    x_s = x_padded.as_strided(
      size=(batch, in_channels, out_height, out_width, *self.kernel_size),
      stride=(x_stride[0], x_stride[1], self.stride[0] * x_stride[-2], self.stride[1] * x_stride[-1], x_stride[-2], x_stride[-1])
    )
    x_s = einops.rearrange(x_s, "batch c_in out_h out_w k_h k_w -> batch 1 c_in out_h out_w k_h k_w")

    w_s = einops.rearrange(self.weight, "c_out c_in k_h k_w -> 1 c_out c_in 1 1 k_h k_w")

    out = x_s * w_s
    assert out.shape == (batch, self.out_channels, self.in_channels, out_height, out_width, *self.kernel_size)

    return einops.reduce(out, "b c_out c_in out_h out_w k_h k_w -> b c_out out_h out_w", "sum")

if __name__ == "__main__":
  layer = Conv2d(2, 3, (2,2), stride=(1,2), padding=(4, 1))
  x = t.arange(0, 36, dtype=t.float32).reshape(2, 2, 3, 3)

  output = layer(x)
  torch_output = F.conv2d(x, layer.weight, stride=(1,2), padding=(4, 1))

  # print("INPUT =========")
  # print(x)
  # print("OUTPUT ========")
  # print(output)
  # print("===========")
  # print(torch_output)

  assert t.allclose(output, torch_output)
  print("Successful!")
