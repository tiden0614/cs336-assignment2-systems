"""End-to-end benchmarking script for the Basics Transformer model.

Subcommands:
    single  -- Benchmark a single (model_size, context_length) config.
    sweep   -- Sweep over multiple model sizes and context lengths.

CLI args (single):
    --model_size     Model config: small | medium | large | xl | 2.7B (default: small)
    --context_length Sequence length (default: 128)
    --mode           forward | both (forward+backward) (default: both)
    --num_warmup     Warmup iterations before timing (default: 5)
    --num_steps      Timed iterations (default: 10)
    --device         cpu | cuda (default: cuda)
    --iter_output    Path to save per-iteration times as CSV (optional, single only)
    --profile        Enable torch.profiler with Python stack traces
    --trace_output   Chrome trace output path (default: trace.json, requires --profile)
    --profile_memory Path to save memory snapshot pickle (for pytorch.org/memory_viz)
    --autocast       Enable BF16 autocast

CLI args (sweep):
    --model_sizes       Space-separated list (default: all)
    --context_lengths   Space-separated list (default: 128 256 512 1024)
    --modes             Space-separated list (default: forward both)
    --num_warmup        (default: 5)
    --num_steps         (default: 10)
    --device            (default: cuda)
    --output            Path to save CSV results (optional)

Example commands:
    # Single config, smoke test on CPU
    uv run python -m cs336_systems.benchmark single --model_size small --context_length 128 --device cpu --num_warmup 1 --num_steps 2

    # Forward-only timing on GPU
    uv run python -m cs336_systems.benchmark single --model_size small --context_length 128 --mode forward

    # Default sweep: all model sizes x [128,256,512,1024] x [forward, both]
    uv run python -m cs336_systems.benchmark sweep

    # Custom sweep with CSV output
    uv run python -m cs336_systems.benchmark sweep --model_sizes small medium --context_lengths 128 256 --modes both --output results.csv

    # Single with per-iteration CSV dump
    uv run python -m cs336_systems.benchmark single --model_size small --context_length 128 --iter_output iters.csv

    # Profile with torch.profiler (outputs Chrome trace with Python stacks)
    uv run nsys profile  \
      --python-sampling=true  \
      --python-backtrace=cuda  \
      --sample=cpu  \
      --cpuctxsw=process-tree  \
      -o result-with-pt-stack2  \
    python -m cs336_systems.benchmark single  \
      --model_size small \
      --context_length 128 \
      --device cuda:0  \
      --num_warmup 10 \
      --num_steps 100 \
      --mode both \
      --profile \
      --trace_output trace.json
    # Then open trace.json in chrome://tracing or https://ui.perfetto.dev
"""

import argparse
import csv
import itertools
import json
import timeit

import math

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import label_backward, profile_range, softmax

MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4


def build_model(model_size: str, context_length: int, device: torch.device) -> BasicsTransformerLM:
    cfg = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        rope_theta=10000.0,
        **cfg,
    )
    return model.to(device)


def make_batch(batch_size: int, context_length: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)


def benchmark(
    model_size: str,
    context_length: int,
    mode: str,
    num_warmup: int,
    num_steps: int,
    device: str,
    trace_output: str | None = None,
    autocast: bool = False,
    profile_memory: str | None = None,
) -> dict[str, float]:
    dev = torch.device(device)
    model = build_model(model_size, context_length, dev)
    if trace_output is not None:
        label_backward(model)
    x = make_batch(BATCH_SIZE, context_length, dev)

    is_cuda = dev.type == "cuda"

    def forward_step():
        if autocast:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                model(x)
        else:
            model(x)
        if is_cuda:
            torch.cuda.synchronize()

    def forward_backward_step():
        if autocast:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = logits.sum()
        else:
            logits = model(x)
            loss = logits.sum()
        loss.backward()
        if is_cuda:
            torch.cuda.synchronize()

    step_fn = forward_step if mode == "forward" else forward_backward_step

    # Warmup
    for _ in range(num_warmup):
        step_fn()
        model.zero_grad(set_to_none=True)

    # Memory profiling — start recording after warmup
    if profile_memory and is_cuda:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Measure
    profiling = trace_output is not None
    prof_ctx = None
    if profiling:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof_ctx = torch.profiler.profile(
            activities=activities,
            with_stack=True,
            record_shapes=True,
        )
        prof_ctx.__enter__()

    timer = timeit.default_timer
    times = []
    for _ in range(num_steps):
        start = timer()
        step_fn()
        elapsed = timer() - start
        times.append(elapsed)
        model.zero_grad(set_to_none=True)

    if profiling:
        prof_ctx.__exit__(None, None, None)

        # Export raw trace, then strip python_function events that cause
        # Perfetto to drop overlapping events (including CUDA kernels).
        raw_trace = trace_output.replace(".json", ".raw.json")
        prof_ctx.export_chrome_trace(raw_trace)

        with open(raw_trace) as f:
            trace_data = json.load(f)

        # Post-process to fix Perfetto rendering:
        # 1. Move python_function events to separate pid (avoids overlapping drops)
        # 2. Move GPU events to a distinct pid with clear naming
        GPU_PID = 999999
        PYTHON_PID = 999998
        cpu_pid = None
        events = []
        for e in trace_data.get("traceEvents", []):
            cat = e.get("cat", "")
            # Move python_function events to their own process
            if cat == "python_function":
                e["pid"] = PYTHON_PID
                events.append(e)
                continue
            # Remap GPU events (pid 0) to a unique pid, drop old metadata
            if e.get("pid") == 0:
                if e.get("ph") == "M":
                    continue  # drop old GPU process metadata
                e["pid"] = GPU_PID
            elif cat == "cpu_op" and cpu_pid is None:
                cpu_pid = e.get("pid")
            # Remap ac2g flow arrows to point to new GPU pid
            if cat == "ac2g" and e.get("ph") == "f":
                e["pid"] = GPU_PID
            events.append(e)

        # Add process metadata
        events.extend([
            {"name": "process_name", "ph": "M", "pid": GPU_PID, "tid": 0,
             "args": {"name": "GPU 0 (CUDA)"}},
            {"name": "process_sort_index", "ph": "M", "pid": GPU_PID, "tid": 0,
             "args": {"sort_index": 0}},
            {"name": "process_name", "ph": "M", "pid": PYTHON_PID, "tid": 0,
             "args": {"name": "Python Stack"}},
            {"name": "process_sort_index", "ph": "M", "pid": PYTHON_PID, "tid": 0,
             "args": {"sort_index": 2}},
        ])
        if cpu_pid is not None:
            events.append(
                {"name": "process_sort_index", "ph": "M", "pid": cpu_pid, "tid": 0,
                 "args": {"sort_index": 1}},
            )

        trace_data["traceEvents"] = events
        with open(trace_output, "w") as f:
            json.dump(trace_data, f)

        print(f"Chrome trace saved to {trace_output} (cleaned for Perfetto)")
        print(f"Raw trace with stacks: {raw_trace}")
        print(f"Open in chrome://tracing or https://ui.perfetto.dev")
        print("\n" + prof_ctx.key_averages(group_by_stack_n=5).table(
            sort_by="cuda_time_total" if is_cuda else "cpu_time_total",
            row_limit=30,
        ))

    # Memory profiling — dump snapshot and stop recording
    if profile_memory and is_cuda:
        torch.cuda.memory._dump_snapshot(profile_memory)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to {profile_memory}")
        print(f"Open at https://pytorch.org/memory_viz")

    times_t = torch.tensor(times)
    return {
        "times": times,
        "mean_s": times_t.mean().item(),
        "std_s": times_t.std().item(),
        "min_s": times_t.min().item(),
        "max_s": times_t.max().item(),
        "p50_s": times_t.median().item(),
        "p95_s": times_t.quantile(0.95).item(),
        "p99_s": times_t.quantile(0.99).item(),
    }


def sweep(
    model_sizes: list[str],
    context_lengths: list[int],
    modes: list[str],
    num_warmup: int,
    num_steps: int,
    device: str,
    output: str | None = None,
    test_autocast: list[bool] = [False],
) -> list[dict]:
    rows = []
    for model_size, context_length, mode, autocast in itertools.product(
        model_sizes, context_lengths, modes, test_autocast
    ):
        ac_label = "bf16" if autocast else "fp32"
        label = f"model={model_size}, ctx={context_length}, mode={mode}, autocast={ac_label}"
        print(f"Running: {label} ...", flush=True)
        try:
            results = benchmark(
                model_size=model_size,
                context_length=context_length,
                mode=mode,
                num_warmup=num_warmup,
                num_steps=num_steps,
                device=device,
                autocast=autocast,
            )
            row = {
                "model_size": model_size,
                "context_length": context_length,
                "mode": mode,
                "autocast": ac_label,
                "mean_ms": results["mean_s"] * 1000,
                "std_ms": results["std_s"] * 1000,
                "min_ms": results["min_s"] * 1000,
                "p50_ms": results["p50_s"] * 1000,
                "p95_ms": results["p95_s"] * 1000,
                "p99_ms": results["p99_s"] * 1000,
                "max_ms": results["max_s"] * 1000,
            }
            print(f"  => p50={row['p50_ms']:.2f}  mean={row['mean_ms']:.2f}  "
                  f"p95={row['p95_ms']:.2f}  min={row['min_ms']:.2f}  max={row['max_ms']:.2f} ms")
        except torch.OutOfMemoryError:
            nan = float("nan")
            row = {
                "model_size": model_size,
                "context_length": context_length,
                "mode": mode,
                "autocast": ac_label,
                "mean_ms": nan, "std_ms": nan, "min_ms": nan,
                "p50_ms": nan, "p95_ms": nan, "p99_ms": nan, "max_ms": nan,
            }
            print("  => OOM")
            torch.cuda.empty_cache()
        rows.append(row)

    # Print summary table
    header = (f"{'model':>8} {'ctx':>6} {'mode':>8} {'cast':>6} "
              f"{'mean':>10} {'std':>10} {'min':>10} {'p50':>10} {'p95':>10} {'p99':>10} {'max':>10}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model_size']:>8} {r['context_length']:>6} {r['mode']:>8} {r['autocast']:>6} "
              f"{r['mean_ms']:>10.2f} {r['std_ms']:>10.2f} {r['min_ms']:>10.2f} "
              f"{r['p50_ms']:>10.2f} {r['p95_ms']:>10.2f} {r['p99_ms']:>10.2f} {r['max_ms']:>10.2f}")

    if output:
        fieldnames = ["model_size", "context_length", "mode", "autocast",
                      "mean_ms", "std_ms", "min_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output}")

    return rows


@profile_range("vanilla_attention")
def vanilla_attention(Q, K, V):
    """Scaled dot-product attention without multihead, no causal mask."""
    d_k = K.shape[-1]
    with profile_range("attention_scores"):
        # autograd saves Q, K
        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    with profile_range("attention_softmax"):
        # autograd saves weights -- the output of the softmax op
        weights = torch.softmax(scores, dim=-1)
    with profile_range("attention_output"):
        # autograd saves weights, V. Since weights is already saved
        # by the last step, autograd will reuse it
        return torch.bmm(weights, V)


def benchmark_attention(
    d_heads: list[int],
    seq_lens: list[int],
    num_warmup: int,
    num_steps: int,
    device: str,
    output: str | None = None,
    profile_memory: str | None = None,
) -> list[dict]:
    """Benchmark attention at various (d_head, seq_len) configurations.

    Per the assignment: batch_size=8, single head, no causal mask.
    Times 100 forward passes, measures memory, then times 100 backward passes.
    """
    BATCH_SIZE_ATTN = 8
    dev = torch.device(device)
    is_cuda = dev.type == "cuda"
    timer = timeit.default_timer
    rows = []

    for d_head, seq_len in itertools.product(d_heads, seq_lens):
        label = f"d_head={d_head}, seq_len={seq_len}"
        print(f"Running: {label} ...", flush=True)

        try:
            Q = torch.randn(BATCH_SIZE_ATTN, seq_len, d_head, device=dev, requires_grad=True)
            K = torch.randn(BATCH_SIZE_ATTN, seq_len, d_head, device=dev, requires_grad=True)
            V = torch.randn(BATCH_SIZE_ATTN, seq_len, d_head, device=dev, requires_grad=True)

            # Warmup forward
            for _ in range(num_warmup):
                out = vanilla_attention(Q, K, V)
                if is_cuda:
                    torch.cuda.synchronize()
                out.detach_()

            # Memory profiling on backward
            if profile_memory and is_cuda:
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            # Time forward passes
            fwd_times = []
            for _ in range(num_steps):
                start = timer()
                out = vanilla_attention(Q, K, V)
                if is_cuda:
                    torch.cuda.synchronize()
                elapsed = timer() - start
                fwd_times.append(elapsed)
                out.detach_()

            # Run one forward to get a live output for backward timing
            out = vanilla_attention(Q, K, V)
            loss = out.sum()

            # Measure memory after forward (includes autograd-saved scores/weights)
            peak_mem_mb = 0.0
            if is_cuda:
                torch.cuda.synchronize()
                peak_mem_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2)  # MB

            # Warmup backward
            for _ in range(num_warmup):
                loss.backward(retain_graph=True)
                if is_cuda:
                    torch.cuda.synchronize()
                Q.grad = K.grad = V.grad = None

            # Reset peak memory tracker before backward timing
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(dev)

            # Time backward passes
            bwd_times = []
            for _ in range(num_steps):
                start = timer()
                loss.backward(retain_graph=True)
                if is_cuda:
                    torch.cuda.synchronize()
                elapsed = timer() - start
                bwd_times.append(elapsed)
                Q.grad = K.grad = V.grad = None

            if profile_memory and is_cuda:
                mem_path = f"{profile_memory}_d{d_head}_s{seq_len}.pickle"
                torch.cuda.memory._dump_snapshot(mem_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                print(f"  Memory snapshot: {mem_path}")

            # Peak memory during backward (high-water mark)
            mem_peak_bwd_mb = 0.0
            if is_cuda:
                torch.cuda.synchronize()
                mem_peak_bwd_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

            fwd_t = torch.tensor(fwd_times)
            bwd_t = torch.tensor(bwd_times)
            row = {
                "d_head": d_head,
                "seq_len": seq_len,
                "fwd_mean_ms": fwd_t.mean().item() * 1000,
                "fwd_std_ms": fwd_t.std().item() * 1000,
                "bwd_mean_ms": bwd_t.mean().item() * 1000,
                "bwd_std_ms": bwd_t.std().item() * 1000,
                "mem_after_fwd_mb": peak_mem_mb,
                "mem_peak_bwd_mb": mem_peak_bwd_mb,
            }
            print(f"  => fwd_mean={row['fwd_mean_ms']:.2f}ms  bwd_mean={row['bwd_mean_ms']:.2f}ms  "
                  f"mem_fwd={row['mem_after_fwd_mb']:.1f}MB  peak_bwd={row['mem_peak_bwd_mb']:.1f}MB")

        except torch.OutOfMemoryError:
            nan = float("nan")
            row = {
                "d_head": d_head,
                "seq_len": seq_len,
                "fwd_mean_ms": nan, "fwd_std_ms": nan,
                "bwd_mean_ms": nan, "bwd_std_ms": nan,
                "mem_after_fwd_mb": nan, "mem_peak_bwd_mb": nan,
            }
            print("  => OOM")
            torch.cuda.empty_cache()

        rows.append(row)

        # Free tensors between configs to avoid OOM cascade
        Q = K = V = out = loss = None  # type: ignore[possibly-undefined]
        if is_cuda:
            torch.cuda.empty_cache()

    # Print summary table
    header = (f"{'d_head':>6} {'seq_len':>8} {'fwd_mean':>10} {'fwd_std':>10} "
              f"{'bwd_mean':>10} {'bwd_std':>10} {'mem_fwd':>10} {'peak_bwd':>10}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['d_head']:>6} {r['seq_len']:>8} {r['fwd_mean_ms']:>10.2f} {r['fwd_std_ms']:>10.2f} "
              f"{r['bwd_mean_ms']:>10.2f} {r['bwd_std_ms']:>10.2f} "
              f"{r['mem_after_fwd_mb']:>10.1f} {r['mem_peak_bwd_mb']:>10.1f}")

    if output:
        fieldnames = list(rows[0].keys())
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output}")

    return rows


def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Basics Transformer model")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- single ---
    p_single = sub.add_parser("single", help="Benchmark a single configuration")
    p_single.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    p_single.add_argument("--context_length", type=int, default=128)
    p_single.add_argument("--mode", type=str, default="both", choices=["forward", "both"])
    p_single.add_argument("--iter_output", type=str, default=None,
                          help="Path to save per-iteration times as CSV")
    p_single.add_argument("--profile", action="store_true",
                          help="Enable torch.profiler with Python stack traces")
    p_single.add_argument("--trace_output", type=str, default="trace.json",
                          help="Chrome trace output path (requires --profile)")
    p_single.add_argument("--profile_memory", type=str, default=None,
                          help="Path to save memory snapshot pickle (for pytorch.org/memory_viz)")
    p_single.add_argument("--autocast", action="store_true",
                          help="Enable BF16 autocast")
    _add_common_args(p_single)

    # --- sweep ---
    p_sweep = sub.add_parser("sweep", help="Sweep over multiple configurations")
    p_sweep.add_argument("--model_sizes", nargs="+", default=list(MODEL_CONFIGS.keys()),
                         choices=list(MODEL_CONFIGS.keys()), metavar="SIZE")
    p_sweep.add_argument("--context_lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    p_sweep.add_argument("--modes", nargs="+", default=["forward", "both"], choices=["forward", "both"])
    p_sweep.add_argument("--output", type=str, default=None, help="Path to save CSV results")
    p_sweep.add_argument("--test_autocast", nargs="+", type=str, default=["false"],
                         choices=["true", "false"],
                         help="Sweep autocast on/off (default: false only)")
    _add_common_args(p_sweep)

    # --- attention ---
    p_attn = sub.add_parser("attention", help="Benchmark vanilla attention at various scales")
    p_attn.add_argument("--d_heads", nargs="+", type=int, default=[16, 32, 64, 128])
    p_attn.add_argument("--seq_lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    p_attn.add_argument("--output", type=str, default=None, help="Path to save CSV results")
    p_attn.add_argument("--profile_memory", type=str, default=None,
                         help="Path prefix for memory snapshots")
    _add_common_args(p_attn)

    args = parser.parse_args()

    if args.command == "single":
        print(f"Config: model={args.model_size}, ctx={args.context_length}, "
              f"mode={args.mode}, warmup={args.num_warmup}, steps={args.num_steps}, device={args.device}")
        results = benchmark(
            model_size=args.model_size,
            context_length=args.context_length,
            mode=args.mode,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            device=args.device,
            trace_output=args.trace_output if args.profile else None,
            autocast=args.autocast,
            profile_memory=args.profile_memory,
        )
        for key in ["mean_s", "std_s", "min_s", "p50_s", "p95_s", "p99_s", "max_s"]:
            label = key.replace("_s", "").upper()
            print(f"  {label:>4}: {results[key]*1000:.2f} ms")

        if args.iter_output:
            with open(args.iter_output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "time_ms"])
                for i, t in enumerate(results["times"]):
                    writer.writerow([i, t * 1000])
            print(f"\nPer-iteration times saved to {args.iter_output}")

    elif args.command == "sweep":
        sweep(
            model_sizes=args.model_sizes,
            context_lengths=args.context_lengths,
            modes=args.modes,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            device=args.device,
            output=args.output,
            test_autocast=[s == "true" for s in args.test_autocast],
        )

    elif args.command == "attention":
        benchmark_attention(
            d_heads=args.d_heads,
            seq_lens=args.seq_lens,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            device=args.device,
            output=args.output,
            profile_memory=args.profile_memory,
        )


if __name__ == "__main__":
    main()
