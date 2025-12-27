import torch as t
import triton

from kernels.vector_add import vector_add

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['size'],
    x_vals=[2**i for i in range(12, 28, 1)],
    x_log=True, # logarithmic
    line_arg='provider',
    line_names=['Triton', 'Torch'],
    line_vals=['triton', 'torch'],
    styles=[('blue', '-'), ('green', '-')],
    ylabel='GB/s',
    plot_name='vector-add-performance',
    args={}
  )
)
def run_vector_add_benchmark(size, provider):
  a = t.rand(size, device=DEVICE, dtype=t.float32)
  b = t.rand(size, device=DEVICE, dtype=t.float32)

  quantiles = [0.5, 0.2, 0.8]

  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: a + b, quantiles=quantiles) # type: ignore
  elif provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add(a, b), quantiles=quantiles) # type: ignore
  else:
    raise ValueError(f"Unknown provider: {provider}")
  gbps = lambda ms: 3*a.numel()*a.element_size()*1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__=="__main__":
  run_vector_add_benchmark.run(print_data=True)
