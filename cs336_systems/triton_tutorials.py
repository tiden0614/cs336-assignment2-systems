import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
    x_ptr, y_ptr, # Input pointer
    output_ptr, # Output pointer
    n_elements, # Size of the vector
    BLOCK_SIZE: tl.constexpr
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here.
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + idx, mask=idx < n_elements)
    y = tl.load(y_ptr + idx, mask=idx < n_elements)
    tl.store(output_ptr + idx, x + y, mask=idx < n_elements)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE, "Tensors must be on the same device as Triton"

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output