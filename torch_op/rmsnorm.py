import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  def __init__(self, normalized_shape, eps=None):
    super().__init__()
    self.normalized_shape = normalized_shape
    self.n = math.prod(self.normalized_shape)
    self.eps = eps

    self.weight = nn.Parameter(t.ones(self.normalized_shape))

  def forward(self, x: t.Tensor):
    ms = x.square().sum(-1) / self.n
    if self.eps:
      rms = t.sqrt(self.eps + ms)
    else:
      rms = t.sqrt(ms)

    return (x/rms) * self.weight


if __name__ == "__main__":
  layer = RMSNorm((3, 3))
  x = t.arange(0, 9, dtype=t.float32).reshape(3, 3)

  output = layer(x)
  torch_output = F.rms_norm(x, (3,3), layer.weight, layer.eps)

  print("INPUT =========")
  print(x)
  print("OUTPUT ========")
  print(output)
  print("===========")
  print(torch_output)

  assert t.allclose(output, torch_output)
  print("Successful!")
