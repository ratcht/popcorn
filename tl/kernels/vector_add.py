import torch as t
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def vector_add_kernel(
  a_ptr,
  b_ptr,
  c_ptr,  # pointers to A, B, C
  n,  # sice of vector
  BLOCK_SIZE: tl.constexpr,  # num elements each program should process. constexpr so it can be used as a shape value
):
  pid = tl.program_id(axis=0)  # 1d launch grid so axis is 0
  # this program process a block that is offset from initial
  # e.g n=256, blocksize=64 -> p0: [0:64], p1: [64:128], p2: [128:192], p3: [192, 256]
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE) # similar to torch.arange

  mask = offsets < n # create mask where True = in bounds

  a = tl.load(a_ptr + offsets, mask=mask)
  b = tl.load(b_ptr + offsets, mask=mask)

  c = a + b

  # write back
  tl.store(c_ptr + offsets, c, mask=mask)

def vector_add(a: t.Tensor, b: t.Tensor):
  c = t.empty_like(a)
  assert a.device == b.device == c.device == DEVICE

  n = c.numel()
  # SPMD launch grid is analogous to cuda launch grids
  # here is a 1d grid where size = num blocks
  grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

  # note, each t.tensor is implicitly converted to a pointer to its first elem
  vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)  # type: ignore[reportArgumentType]

  return c


if __name__=="__main__":
  t.manual_seed(0)

  size = 98432
  a = t.rand(size, device=DEVICE)
  b = t.rand(size, device=DEVICE)
  c = vector_add(a, b)
  c_ref = a + b

  print(c)
  print(c_ref)
  print(t.allclose(c, c_ref))

  print(f'The maximum difference between torch and triton is '
      f'{t.max(t.abs(c_ref - c))}')
