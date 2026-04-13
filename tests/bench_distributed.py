"""Benchmark harness for single-node all-reduce across world sizes and data sizes.

Usage:
    uv run python -m cs336_systems.bench_distributed [--backend gloo] [--num_warmup 3] [--num_steps 10] [--output results.csv]
"""

import os
import csv
import argparse
import time
import itertools
import matplotlib
import timeit

from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import matplotlib.pyplot as plt
from cs336_systems.distributed import DDPIndividualParameters, GradBucket

from cs336_basics.model import BasicsTransformerLM

from .adapters import (
    ddp_individual_parameters_on_after_backward,
    get_ddp_individual_parameters,
)
from .common import (
    _cleanup_process_group,
    _setup_process_group,
)

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4


def _worker(
    local_rank,
    local_world_size,
    data_size_bytes,
    backend,
    num_warmup,
    num_steps,
    result_dict,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=local_rank, world_size=local_world_size)

    dev = "cpu" if backend == "gloo" else f"cuda:{local_rank}"
    numel = data_size_bytes // 4  # float32
    data = torch.rand(numel, dtype=torch.float32, device=dev)

    # Warmup
    for _ in range(num_warmup):
        dist.all_reduce(data, async_op=False)

    # Timed iterations
    if dev != "cpu":
        torch.cuda.synchronize(dev)
    dist.barrier()

    times = []
    for _ in range(num_steps):
        start = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        if dev != "cpu":
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    dist.destroy_process_group()

    # Only rank 0 reports
    if local_rank == 0:
        result_dict["times"] = times


def run_single_config(world_size, data_size_bytes, backend, num_warmup, num_steps):
    """Spawn workers and return timing results from rank 0."""
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _worker,
        args=(world_size, data_size_bytes, backend, num_warmup, num_steps, result_dict),
        nprocs=world_size,
        join=True,
    )

    times = list(result_dict["times"])
    t = torch.tensor(times)
    return {
        "mean_s": t.mean().item(),
        "std_s": t.std().item(),
        "min_s": t.min().item(),
        "times": times,
    }


def parse_size(s: str) -> int:
    """Parse human-readable size like '1MB', '100MB', '1GB' into bytes."""
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def format_size(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.0f}GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.0f}MB"
    elif nbytes >= 1024:
        return f"{nbytes / 1024:.0f}KB"
    return f"{nbytes}B"


def benchmar_single_node(
    data_sizes: list[str],
    world_sizes: list[int],
    backend: str,
    num_warmup: int,
    num_steps: int,
    output: str,
    chart: bool,
):
    data_size_bytes = [parse_size(s) for s in data_sizes]
    rows = []

    for ws, ds_bytes in itertools.product(world_sizes, data_size_bytes):
        ds_label = format_size(ds_bytes)
        print(
            f"Running: world_size={ws}, data_size={ds_label}, backend={backend} ...",
            flush=True,
        )

        results = run_single_config(ws, ds_bytes, backend, num_warmup, num_steps)

        throughput_mbs = (ds_bytes / results["mean_s"]) / (1024**2)
        row = {
            "world_size": ws,
            "data_size": ds_label,
            "data_size_bytes": ds_bytes,
            "mean_ms": results["mean_s"] * 1000,
            "std_ms": results["std_s"] * 1000,
            "min_ms": results["min_s"] * 1000,
            "throughput_mbs": throughput_mbs,
        }
        rows.append(row)
        print(
            f"  => mean={row['mean_ms']:.2f}ms  std={row['std_ms']:.2f}ms  "
            f"throughput={throughput_mbs:.1f} MB/s"
        )

    # Print summary table
    header = (
        f"{'ws':>4} {'data':>8} {'mean_ms':>10} {'std_ms':>10} "
        f"{'min_ms':>10} {'tput_MB/s':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['world_size']:>4} {r['data_size']:>8} {r['mean_ms']:>10.2f} {r['std_ms']:>10.2f} "
            f"{r['min_ms']:>10.2f} {r['throughput_mbs']:>12.1f}"
        )

    if output:
        fieldnames = [
            "world_size",
            "data_size",
            "data_size_bytes",
            "mean_ms",
            "std_ms",
            "min_ms",
            "throughput_mbs",
        ]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output}")

    if chart:
        _draw_chart(rows, chart, world_sizes, data_sizes)


def _draw_chart(rows, chart_path, world_sizes, data_size_labels):
    matplotlib.use("Agg")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Group by world_size
    for ws in world_sizes:
        ws_rows = [r for r in rows if r["world_size"] == ws]
        sizes = [r["data_size"] for r in ws_rows]
        times = [r["mean_ms"] for r in ws_rows]
        tputs = [r["throughput_mbs"] for r in ws_rows]

        ax1.plot(sizes, times, "o-", label=f"ws={ws}")
        ax2.plot(sizes, tputs, "o-", label=f"ws={ws}")

    ax1.set_xlabel("Data Size")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("All-Reduce Latency")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Data Size")
    ax2.set_ylabel("Throughput (MB/s)")
    ax2.set_title("All-Reduce Throughput")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")


@dataclass
class ParamCommSample:
    name: str
    numel: int
    time_ms: float


@dataclass
class NaiveDDPBenchmarkIterationSample:
    fwd_ms: float = 0.0
    bwd_ms: float = 0.0
    comm_ms: float = 0.0
    comm_cnt: int = 0
    param_comms: list[ParamCommSample] | None = None


class measure:
    """Context manager that records elapsed time in milliseconds to a list."""

    def __init__(self, collector: list[float]):
        self.col = collector
        self.start = 0.0

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (timeit.default_timer() - self.start) * 1000
        self.col.append(elapsed_ms)


NO_BATCH = 0
BATCH_ALL = 1
BUCKETED_ASYNC = 2


def benchmark_naive_ddp(
    model_sizes: list[str],
    world_sizes: list[int],
    num_warmup: int,
    num_steps: int,
    context_lengths: list[int],
    comm_mode: int = NO_BATCH,
):
    manager = mp.Manager()
    for model_size, context_length, world_size in itertools.product(
        model_sizes, context_lengths, world_sizes
    ):
        result_dict = manager.dict()
        mp.spawn(
            naive_ddp_worker,
            args=(
                world_size,
                num_warmup,
                num_steps,
                model_size,
                context_length,
                comm_mode,
                result_dict,
            ),
            nprocs=world_size,
            join=True,
        )
        # Print results from rank 0
        samples = list(result_dict["samples"])
        print(
            f"\n=== Naive DDP  model={model_size}  ctx={context_length}  "
            f"world_size={world_size}  warmup={num_warmup}  steps={num_steps} ==="
        )
        print(
            f"{'step':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'comm_ms':>10} {'comm_cnt':>10}"
        )
        print("-" * 48)
        for i, s in enumerate(samples):
            print(
                f"{i:>6} {s.fwd_ms:>10.3f} {s.bwd_ms:>10.3f} "
                f"{s.comm_ms:>10.3f} {s.comm_cnt:>10}"
            )

        fwd = torch.tensor([s.fwd_ms for s in samples])
        bwd = torch.tensor([s.bwd_ms for s in samples])
        comm = torch.tensor([s.comm_ms for s in samples])
        print(f"\n{'':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'comm_ms':>10}")
        print(
            f"{'mean':>6} {fwd.mean():>10.3f} {bwd.mean():>10.3f} {comm.mean():>10.3f}"
        )
        print(f"{'std':>6} {fwd.std():>10.3f} {bwd.std():>10.3f} {comm.std():>10.3f}")

        # Per-parameter comm breakdown: average across steps, sorted by numel
        if samples and samples[0].param_comms:
            # Aggregate: for each param, average time across steps
            from collections import defaultdict

            param_times: dict[str, list[float]] = defaultdict(list)
            param_numel: dict[str, int] = {}
            for s in samples:
                for pc in s.param_comms:
                    param_times[pc.name].append(pc.time_ms)
                    param_numel[pc.name] = pc.numel

            print(
                f"\n  Per-parameter comm (averaged over {len(samples)} steps, sorted by numel):"
            )
            print(
                f"  {'numel':>12} {'size_KB':>10} {'mean_ms':>10} {'std_ms':>10}  {'name'}"
            )
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}  {'-'*40}")

            sorted_params = sorted(param_numel.items(), key=lambda x: x[1])
            total_numel = 0
            total_time_ms = 0.0
            for name, numel in sorted_params:
                times_t = torch.tensor(param_times[name])
                size_kb = numel * 4 / 1024  # float32
                mean_ms = times_t.mean().item()
                std_ms = times_t.std().item() if len(times_t) > 1 else 0.0
                total_numel += numel
                total_time_ms += mean_ms
                print(
                    f"  {numel:>12} {size_kb:>10.1f} {mean_ms:>10.3f} {std_ms:>10.3f}  {name}"
                )

            print(f"  {'-'*12} {'-'*10} {'-'*10}")
            print(
                f"  {total_numel:>12} {total_numel*4/1024:>10.1f} {total_time_ms:>10.3f}  TOTAL ({len(sorted_params)} params)"
            )


def naive_ddp_worker(
    rank: int,
    world_size: int,
    num_warmup: int,
    num_steps: int,
    model_size: str,
    context_length: int,
    comm_mode: int, # NO_BATCH, BATCH_ALL, BUCKETED_ASYNC
    result_dict,
):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(rank)
    cfg = MODEL_CONFIGS[model_size]
    model_base = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        rope_theta=10000.0,
        **cfg,
    ).to(device)
    model_rank = DDPIndividualParameters(model_base, async_all_reduce=comm_mode == BUCKETED_ASYNC)

    loss_fn = nn.CrossEntropyLoss()
    ddp_optimizer = optim.SGD(model_rank.parameters(), lr=1e-3)

    # Each rank gets its own random batch of token IDs per step
    local_bs = BATCH_SIZE // world_size or 1

    samples: list[NaiveDDPBenchmarkIterationSample] = []

    for i in range(num_warmup + num_steps):
        ddp_optimizer.zero_grad()

        # Random token IDs: input is [0, context_length-1), target is [1, context_length)
        tokens = torch.randint(0, VOCAB_SIZE, (local_bs, context_length), device=device)
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]

        fwd_times: list[float] = []
        with measure(fwd_times):
            logits = model_rank(input_ids)  # (bs, seq_len-1, vocab_size)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))

        # For BUCKETED_ASYNC, all-reduces fire during backward via hooks,
        # so we measure bwd+comm together and then the wait/residual separately.
        bwd_times: list[float] = []
        with measure(bwd_times):
            loss.backward()

        world_sz = dist.get_world_size()
        param_comms: list[ParamCommSample] = []
        comm_start = timeit.default_timer()

        if comm_mode == BATCH_ALL:
            bucket = GradBucket()
            for _, param in model_rank.named_parameters():
                if param.grad is not None:
                    bucket.add(param.grad.data)

            flat = bucket.flatten()
            t0 = timeit.default_timer()
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            t1 = timeit.default_timer()
            bucket.scatter_back()

            param_comms.append(ParamCommSample(
                name=f"flat_bucket({len(bucket._grads)} params)",
                numel=bucket.numel,
                time_ms=(t1 - t0) * 1000,
            ))

        elif comm_mode == BUCKETED_ASYNC:
            # Hooks already fired async all-reduces during backward.
            # finish_gradient_synchronization flushes residual bucket + waits.
            n_buckets_before = len(model_rank._async_work_handles)
            model_rank.finish_gradient_synchronization()
            n_buckets_total = n_buckets_before + 1  # +1 for residual flushed in finish
            total_numel = sum(p.numel() for p in model_rank.parameters() if p.grad is not None)
            param_comms.append(ParamCommSample(
                name=f"bucketed_async({n_buckets_total} buckets)",
                numel=total_numel,
                time_ms=0.0,  # filled below
            ))

        else:  # NO_BATCH: per-parameter
            for name, param in model_rank.named_parameters():
                if param.grad is not None:
                    t0 = timeit.default_timer()
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_sz
                    t1 = timeit.default_timer()
                    param_comms.append(
                        ParamCommSample(
                            name=name,
                            numel=param.numel(),
                            time_ms=(t1 - t0) * 1000,
                        )
                    )

        comm_total_ms = (timeit.default_timer() - comm_start) * 1000
        # For BUCKETED_ASYNC, comm_total_ms is just the wait/residual time
        if comm_mode == BUCKETED_ASYNC and param_comms:
            param_comms[0].time_ms = comm_total_ms

        ddp_optimizer.step()

        if i >= num_warmup:
            samples.append(
                NaiveDDPBenchmarkIterationSample(
                    fwd_ms=fwd_times[0],
                    bwd_ms=bwd_times[0],
                    comm_ms=comm_total_ms,
                    comm_cnt=len(param_comms),
                    param_comms=param_comms,
                )
            )

    if rank == 0:
        result_dict["samples"] = samples

    _cleanup_process_group()


def main():
    parser = argparse.ArgumentParser(description="Benchmark distributed training")
    sub = parser.add_subparsers(dest="command", required=True)

    p_single_node = sub.add_parser(
        "single-node", help="Benchmark single-node distributed"
    )
    p_single_node.add_argument("--world_sizes", nargs="+", type=int, default=[2, 4, 6])
    p_single_node.add_argument(
        "--data_sizes", nargs="+", type=str, default=["1MB", "10MB", "100MB", "1GB"]
    )
    p_single_node.add_argument(
        "--backend", type=str, default="gloo", choices=["gloo", "nccl"]
    )
    p_single_node.add_argument("--num_warmup", type=int, default=3)
    p_single_node.add_argument("--num_steps", type=int, default=10)
    p_single_node.add_argument(
        "--output", type=str, default=None, help="Path to save CSV results"
    )
    p_single_node.add_argument(
        "--chart", type=str, default=None, help="Path to save chart PNG"
    )

    p_naive_ddp = sub.add_parser("naive-ddp", help="Benchmark naive ddp")
    p_naive_ddp.add_argument(
        "--model_sizes",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        metavar="SIZE",
    )
    p_naive_ddp.add_argument("--context_lengths", nargs="+", type=int, default=[256])
    p_naive_ddp.add_argument("--world_sizes", nargs="+", type=int, default=[2])
    p_naive_ddp.add_argument("--num_warmup", type=int, default=3)
    p_naive_ddp.add_argument("--num_steps", type=int, default=5)
    p_naive_ddp.add_argument("--comm_mode", type=int, default=NO_BATCH,
                             choices=[NO_BATCH, BATCH_ALL, BUCKETED_ASYNC],
                             help=f"0=per-param, 1=flat all, 2=bucketed async")

    args = parser.parse_args()

    if args.command == "single-node":
        benchmar_single_node(
            args.data_sizes,
            args.world_sizes,
            args.backend,
            args.num_warmup,
            args.num_steps,
            args.output,
            args.chart,
        )
    elif args.command == "naive-ddp":
        benchmark_naive_ddp(
            args.model_sizes,
            args.world_sizes,
            args.num_warmup,
            args.num_steps,
            args.context_lengths,
            args.comm_mode,
        )


if __name__ == "__main__":
    main()
