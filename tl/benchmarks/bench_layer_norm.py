import torch as t
import triton

from kernels.layer_norm import layer_norm_fwd, layer_norm_fwd_naive

DEVICE = triton.runtime.driver.active.get_active_torch_device()
TORCH_HAS_FP8 = hasattr(t, "float8_e5m2")

def is_cuda():
  return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['N'],
    x_vals=[512*i for i in range(2,32)],
    line_arg='provider',
    line_names=['Triton', 'Torch', 'Naive'],
    line_vals=['triton', 'torch', 'naive'],
    styles=[('blue', '-'), ('green', '-'), ('red', '-')],
    ylabel='GB/s',
    plot_name='layer_norm-performance',
    args={'M': 4096, 'dtype': t.float16}
  )
)
def run_layer_norm_benchmark(M, N, dtype, provider, eps=1e-5, device=DEVICE):
  x = t.rand((M, N), device=DEVICE, dtype=dtype)
  normalized_shape = (N,)
  weight = t.rand(normalized_shape, device=DEVICE, dtype=dtype)*2 - 1
  bias = t.rand(normalized_shape, device=DEVICE, dtype=dtype)*2

  quantiles = [0.5, 0.2, 0.8]

  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: t.layer_norm(x, normalized_shape, weight, bias, eps), quantiles=quantiles) # type: ignore
  elif provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: layer_norm_fwd(x, normalized_shape, weight, bias, eps), quantiles=quantiles) # type: ignore
  elif provider == 'naive':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: layer_norm_fwd_naive(x, normalized_shape, weight, bias, eps), quantiles=quantiles) # type: ignore
  else:
    raise ValueError(f"Unknown provider: {provider}")
  gbps = lambda ms: 2*x.numel()*x.element_size()*1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__=="__main__":
  print(f"Device: {DEVICE}")
  run_layer_norm_benchmark.run(print_data=True, save_path='.')
