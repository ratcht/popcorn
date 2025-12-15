import torch as t
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
  def __init__(self, normalized_shape, eps=1e-05):
    super().__init__()
    self.normalized_shape = normalized_shape
    self.dims = tuple(-i - 1 for i in range(len(self.normalized_shape)))
    self.eps = eps

    self.weight = nn.Parameter(t.ones(self.normalized_shape))
    self.bias = nn.Parameter(t.zeros(self.normalized_shape))

  def forward(self, x: t.Tensor):
    e_x = x.mean(
      self.dims,
      keepdim=True
    )
    var_x = x.var(
      self.dims,
      keepdim=True,
      correction=0
    )

    return (
      (x - e_x)/t.sqrt(var_x + self.eps)
    ) * self.weight + self.bias


if __name__ == "__main__":
  layer = LayerNorm((3, 3))
  x = t.arange(0, 9, dtype=t.float32).reshape(3, 3)

  output = layer(x)
  torch_output = F.layer_norm(x, (3,3), layer.weight, layer.bias, layer.eps)

  print("INPUT =========")
  print(x)
  print("OUTPUT ========")
  print(output)
  print("===========")
  print(torch_output)

  assert t.allclose(output, torch_output)
  print("Successful!")
