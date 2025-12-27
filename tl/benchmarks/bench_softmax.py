import torch as t
import triton

from ..kernels.softmax import softmax


def softmax_naive(x):
  """row wise softmax
  softmax(x_i) = e^x_i / sum(e^x)
  """
  # logsumexp
  x_max = x.max(dim=-1, keepdim=True).values

  z = x - x_max

  p = t.exp(z)
  q = p.sum(dim=-1, keepdim=True)

  return p / q


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['N'],
    x_vals=[128*i for i in range(2,100)],
    x_log=True, # logarithmic
    line_arg='provider',
    line_names=['Triton', 'Torch', 'Naive'],
    line_vals=['triton', 'torch', 'naive'],
    styles=[('blue', '-'), ('green', '-'), ('red', '-')],
    ylabel='GB/s',
    plot_name='softmax-performance',
    args={'M': 4096}
  )
)
def run_softmax_benchmark(M, N, provider):
  x = t.rand((M, N), device=DEVICE, dtype=t.float32)

  quantiles = [0.5, 0.2, 0.8]

  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: t.softmax(x, axis=-1), quantiles=quantiles) # type: ignore
  elif provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles) # type: ignore
  elif provider == 'naive':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_naive(x), quantiles=quantiles) # type: ignore
  else:
    raise ValueError(f"Unknown provider: {provider}")
  gbps = lambda ms: 2*x.numel()*x.element_size()*1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__=="__main__":
  print(f"Device: {DEVICE}")
  run_softmax_benchmark.run(print_data=True, save_path='.')
