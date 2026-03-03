import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import record_function


class profile_range:
    """Combined NVTX range + torch.profiler record_function.

    Works as both a context manager and a decorator.
    Shows up in both Nsight Systems (NVTX row) and torch.profiler Chrome traces.

    Usage:
        @profile_range("my_func")
        def my_func(...): ...

        with profile_range("my_block"):
            ...
    """

    def __init__(self, label: str):
        self.label = label
        # Pre-build the decorator chain so the decorated function itself
        # is the call site (not wrapper code in this file), giving
        # torch.profiler correct stack attribution.
        self._nvtx_decorator = nvtx.range(label)
        self._rf_decorator = record_function(label)

    # -- context manager --
    def __enter__(self):
        self._nvtx_decorator.__enter__()
        self._rf_decorator.__enter__()
        return self

    def __exit__(self, *args):
        self._rf_decorator.__exit__(*args)
        self._nvtx_decorator.__exit__(*args)

    # -- decorator --
    def __call__(self, fn):
        # Stack nvtx and record_function as decorators directly.
        # record_function.__call__ and nvtx.range.__call__ each wrap fn,
        # so the profiler sees the decorated function's call site, not ours.
        return self._nvtx_decorator(self._rf_decorator(fn))


def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(inputs, targets):
    negative_log_softmax_logits = -log_softmax(inputs)
    return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))


def clip_gradient(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = 0.0

    for g in grads:
        norm += (g**2).sum()

    norm = torch.sqrt(norm)
    clip_coef = min(1, max_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef
