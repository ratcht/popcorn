# pyright: reportGeneralTypeIssues=false, reportUnreachable=false, reportAttributeAccessIssue=false, reportIndexIssue=false
import torch as t
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def softmax_kernel(in_ptr, out_ptr, in_row_stride, out_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
  row_start = tl.program_id(0)
  row_step = tl.num_programs(0)
  # note, we must pad each row as each block must be 2^x

  for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    row_start_ptr = in_ptr + row_idx*in_row_stride # stride represents amnt to advance by 1 row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_ptrs = row_start_ptr + col_offsets

    # load into sram using mask
    mask = col_offsets < n_cols
    row = tl.load(in_ptrs, mask=mask, other=-float('inf'))

    # subtract max
    z = row - tl.max(row, axis=0)

    p = tl.exp(z)
    q = tl.sum(p, axis=0)
    out = p / q

    out_row_start_ptr = out_ptr + row_idx*out_row_stride
    out_ptrs = out_row_start_ptr + col_offsets
    tl.store(out_ptrs, out, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x):
  n_rows, n_cols = x.shape

  BLOCK_SIZE = triton.next_power_of_2(n_cols)

  num_warps = 8
  num_stages = 4 if SIZE_SMEM > 200000 else 2

  y = t.empty_like(x)

  # pre-compile kernel to get register usage and compute thread occupany
  kernel = softmax_kernel.warmup(x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                num_stages=num_stages, num_warps=num_warps, grid=(1, ))

  kernel._init_handles()

  n_regs = kernel.n_regs
  size_smem = kernel.metadata.shared

  occupancy = min(NUM_REGS // (n_regs * WARP_SIZE * num_warps), SIZE_SMEM // size_smem)
  num_programs = min(NUM_SM * occupancy, n_rows)

  # Create a number of persistent programs.
  kernel[(num_programs, 1, 1)](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
  return y

if __name__ == "__main__":
  t.manual_seed(0)
  x = t.rand((6, 4), dtype=t.float32, device=DEVICE)
  print(x.shape)
  print(x)
  y_1 = softmax(x)
  y_2 = t.softmax(x, dim=-1)

  print("Triton output:")
  print(y_1)
  print("\nPyTorch output:")
  print(y_2)
  print("\nDifference:")
  print(y_1 - y_2)

  assert t.allclose(y_1, y_2)

  print("success!")
