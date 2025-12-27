# pyright: reportGeneralTypeIssues=false, reportUnreachable=false, reportAttributeAccessIssue=false, reportIndexIssue=false
import math

import torch as t
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def layer_norm_fwd_naive(x: t.Tensor, normalized_shape, weight, bias=None, eps=1e-05):
  dims = tuple(range(-len(normalized_shape), 0))
  e_x = x.mean(dims, keepdim=True)
  var_x = x.var(dims, keepdim=True, correction=0)

  p = x - e_x
  q = t.sqrt(var_x + eps)

  out = (p/q) * weight

  if bias is not None:
    out += bias

  return out

@triton.jit
def layer_norm_fwd_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, r_std_ptr, stride, n, eps, BLOCK_SIZE: tl.constexpr):
  start = tl.program_id(0)

  # (*, 4, 5, 3) -> n = 60. must jump 60 elems between programs

  x_ptr += start * stride
  y_ptr += start * stride

  # calculate E[x]. chunked
  _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
  for off in range(0, n, BLOCK_SIZE):
    offsets = off + tl.arange(0, BLOCK_SIZE)
    x_chunk = tl.load(x_ptr + offsets, mask=offsets < n, other=0.).to(tl.float32)
    _mean += x_chunk
  mean = tl.sum(_mean) / n

  # calculate Var[x]. chunked
  _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
  for off in range(0, n, BLOCK_SIZE):
    offsets = off + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x_chunk = tl.load(x_ptr + offsets, mask=mask, other=0.).to(tl.float32)
    x_chunk = tl.where(mask, x_chunk - mean, 0.0)
    _var += x_chunk * x_chunk
  var = tl.sum(_var) / n

  # store mean and r_std
  r_std = 1 / tl.sqrt(var + eps)

  tl.store(mean_ptr + start, mean)
  tl.store(r_std_ptr + start, r_std)

  # apply weights
  for off in range(0, n, BLOCK_SIZE):
    offsets = off + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    y = ((x - mean) * r_std) * w + b

    tl.store(y_ptr + offsets, y, mask=mask)


def layer_norm_fwd(x: t.Tensor, normalized_shape, weight: t.Tensor, bias: t.Tensor, eps=1e-05):
  y = t.empty_like(x)
  assert x.device == y.device == DEVICE

  # reshape to 2D: (*, *normalized_shape) -> (M, N)
  n = math.prod(normalized_shape)
  x_arg = x.reshape(-1, n)
  M, N = x_arg.shape

  # allocate mean and rstd
  mean = t.empty((M,), dtype=t.float32, device=x.device)
  r_std = t.empty((M,), dtype=t.float32, device=x.device)

  # heuristics
  MAX_FUSED_SIZE = 65536 // x.element_size() # assume shmem limit = 64kb
  BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

  num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

  layer_norm_fwd_kernel[(M,)](
    x_arg, y, weight, bias, mean, r_std,
    x_arg.stride(0), N, eps,
    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1  # type: ignore[reportCallIssue]
  )

  return y


if __name__ == "__main__":
  shape = (5, 10)
  x = t.rand((10, *shape), device=DEVICE)

  weight = t.rand(shape, device=DEVICE)*2 - 1
  bias = t.rand(shape, device=DEVICE)*2

  y_ref = t.layer_norm(x, shape, weight, bias, eps=1e-05)
  y_kernel = layer_norm_fwd(x, shape, weight, bias, eps=1e-05)
  y_naive = layer_norm_fwd_naive(x, shape, weight, bias)

  print(f"ref: {y_ref}")
  print(f"triton: {y_kernel}")

  assert t.allclose(y_ref, y_naive)
  print('success ref vs. naive!')

  assert t.allclose(y_ref, y_kernel)
  print('success ref vs. kernel!')
