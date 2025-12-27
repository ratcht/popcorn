import torch as t
import triton

from kernels.matmul import matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()
TORCH_HAS_FP8 = hasattr(t, "float8_e5m2")

def is_cuda():
  return triton.runtime.driver.active.get_current_target().backend == "cuda"


configs=[]
for fp8_inputs in [False, True]:
  if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
    continue
  configs.append(
    triton.testing.Benchmark(
      x_names=['M', 'N', 'K'],
      x_vals=[128*i for i in range(2,33)],
      line_arg='provider',
      line_names=['Triton'] if fp8_inputs else ['Torch', 'Triton'],
      line_vals=['triton'] if fp8_inputs else ['torch', 'triton'],
      styles=[('green', '-'), ('red', '-')],
      ylabel='TFLOPS',
      plot_name='matmul-performance-'+('fp16' if not fp8_inputs else 'fp8'),
      args={'fp8_inputs': fp8_inputs}
    )
  )

@triton.testing.perf_report(configs)
def run_matmul_benchmark(M, N, K, provider, fp8_inputs):
  a = t.rand((M, K), device=DEVICE, dtype=t.float16)
  b = t.rand((K, N), device=DEVICE, dtype=t.float16)

  if TORCH_HAS_FP8 and fp8_inputs:
    a = a.to(t.float8_e5m2)
    b = b.T
    b = b.to(t.float8_e5m2)

  quantiles = [0.5, 0.2, 0.8]

  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: t.matmul(a, b), quantiles=quantiles) # type: ignore
  elif provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles) # type: ignore
  else:
    raise ValueError(f"Unknown provider: {provider}")
  perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
  return perf(ms), perf(max_ms), perf(min_ms)


if __name__=="__main__":
  print(f"Device: {DEVICE}")
  run_matmul_benchmark.run(print_data=True, save_path='.')
