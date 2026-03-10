import torch
import triton
import cs336_systems.triton_tutorials as triton_tutorials
import pytest

def test_add():
    torch.manual_seed(0)
    size = 10000

    for _ in range(100):
        x = torch.rand(size, device=triton_tutorials.DEVICE)
        y = torch.rand(size, device=triton_tutorials.DEVICE)
        output_torch = x + y
        output_triton = triton_tutorials.add(x, y)
        assert torch.allclose(output_torch, output_triton), "Triton add does not match PyTorch add"
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={}
    ))
def benchmark_triton_add(size, provider):
    x = torch.rand(size, device=triton_tutorials.DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=triton_tutorials.DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_tutorials.add(x, y), quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


@pytest.mark.benchmark(group="triton_add")
def test_benchmark_triton_add():
    benchmark_triton_add.run(print_data=True, show_plots=True)