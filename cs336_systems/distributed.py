import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import logging

log = logging.getLogger(__name__)


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def distributed_demo(rank, world_size):
    setup(rank, world_size, "gloo")
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")


def dist_comm_single_node(
    local_rank, local_world_size, data_size, backend="gloo"
) -> float:
    setup(local_rank, local_world_size, backend)
    dev = "cpu" if backend == "gloo" else f"cuda:{local_rank}"
    numel = data_size // 4  # f32
    data = torch.rand(numel, dtype=torch.float32, device=dev)

    print(f"rank {local_rank} data (before all-reduce): {data}")
    now = time.time()
    dist.all_reduce(data, async_op=False)
    elapsed = time.time() - now
    print(
        f"rank {local_rank} data (after all-reduce): {data}, elapsed={elapsed}, throughput={(data_size / elapsed) / (2 ** 20)} MB/s"
    )

    return elapsed


class GradBucket:
    """Collects grad tensors, flattens into a single buffer for one all-reduce, then scatters back."""

    def __init__(self, name: str = "all"):
        self.name = name
        self._grads: list[torch.Tensor] = []  # references to param.grad.data
        self._numels: list[int] = []
        self._flat: torch.Tensor | None = None

    @property
    def numel(self):
        return sum(self._numels)

    @property
    def size_bytes(self):
        """Estimated size in bytes (assumes float32 until flatten is called)."""
        if self._grads:
            return self.numel * self._grads[0].element_size()
        return self.numel * 4  # default float32

    def add(self, grad: torch.Tensor):
        self._grads.append(grad)
        self._numels.append(grad.numel())

    def flatten(self):
        """Cat all grads into a contiguous flat buffer."""
        self._flat = torch.cat([g.reshape(-1) for g in self._grads])
        return self._flat

    def scatter_back(self):
        """Split the reduced flat buffer and copy back into original grad tensors."""
        if self._flat is None:
            return

        offset = 0
        for grad, n in zip(self._grads, self._numels):
            grad.copy_(self._flat[offset : offset + n].reshape(grad.shape))
            offset += n


class DDPIndividualParameters(torch.nn.Module):
    """Naive DDP: broadcast params at init, all-reduce each grad after backward."""

    def __init__(
        self,
        module: torch.nn.Module,
        async_all_reduce: bool = True,
        bucket_size: int = 100 << 20,  # 100MB
    ):
        super().__init__()
        self.module = module
        self._broadcast_params()
        self._async_work_handles: list[tuple[GradBucket, dist.Work]] = []
        self._bucket = GradBucket()
        self._bucket_size = bucket_size
        if async_all_reduce:
            def all_reduce_hook(p: torch.Tensor):
                self._bucket.add(p.grad)
                if self._bucket.size_bytes >= self._bucket_size:
                    self._flush_bucket()

            for param in self.module.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(all_reduce_hook)

    def _broadcast_params(self):
        rank = dist.get_rank()
        log.info(f"[{rank}] Broadcasting parameters from rank 0")
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        log.info(f"[{rank}] Parameter broadcast complete")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)

    def _flush_bucket(self):
        """Flatten current bucket, launch async all-reduce, stash handle."""
        if not self._bucket._grads:
            return
        work = dist.all_reduce(
            self._bucket.flatten(),
            op=dist.ReduceOp.AVG,
            async_op=True,
        )
        self._async_work_handles.append((self._bucket, work))
        self._bucket = GradBucket()

    def finish_gradient_synchronization(self):
        # Flush any residual grads that didn't fill a full bucket
        self._flush_bucket()
        for bucket, work in self._async_work_handles:
            work.wait()
            bucket.scatter_back()
        self._async_work_handles.clear()


def broadcast_model_param(module: torch.nn.Module) -> DDPIndividualParameters:
    return DDPIndividualParameters(module)


def all_reduce_gradiens(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    # world_size = dist.get_world_size()
    # for param in ddp_model.parameters():
    #     if param.grad is not None:
    #         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    #         param.grad.data /= world_size
    ddp_model.finish_gradient_synchronization()


if __name__ == "__main__":
    local_world_size = 4
    data_size = 1024 << 20  # 1GB
    mp.spawn(
        dist_comm_single_node,
        args=(
            local_world_size,
            data_size,
        ),
        nprocs=local_world_size,
        join=True,
    )
