# CS336 Assignment 2: Systems and Parallelism

## Project Purpose
Self-study of Stanford CS336 (Spring 2025) to learn AI training/inference systems primitives.
**Not** a class submission — the goal is deep learning of concepts, not grades.

## Learner Profile
Staff-level SWE at Meta, 10 years distributed systems experience. Skip boilerplate/setup help.
Focus on: GPU kernel programming (Triton), FlashAttention internals, DDP mechanics, optimizer sharding.

## Project Structure
- `cs336-basics/` — Staff solution for Assignment 1 (Transformer model, AdamW, data loading)
- `cs336_systems/` — **Our implementation directory** (currently empty `__init__.py`)
- `tests/` — Test suite with `adapters.py` as the interface between tests and our code
- `tests/adapters.py` — **Must implement** all adapter functions to connect our code to tests
- Assignment PDF: `cs336_spring2025_assignment2_systems.pdf`

## How to Run
- **Always use**: `uv run python <script>` or `uv run pytest <test>`
- **Run tests**: `uv run pytest tests/<test_file>.py` or `uv run pytest -k <test_name>`
- **Profile with nsys**: `uv run nsys profile -o result python <script>.py`
- Python >=3.11,<3.13; torch; triton; einops; wandb

## Model Sizes (Table 1 from assignment)
| Size   | d_model | d_ff  | num_layers | num_heads |
|--------|---------|-------|------------|-----------|
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 1600    | 6400  | 48         | 25        |
| 2.7B   | 2560    | 10240 | 32         | 32        |

All models: vocab_size=10000, batch_size=4.

## Assignment Problems (ordered by section)

### Part 1: Single-GPU Optimization
1. **benchmarking_script** (4pt) — End-to-end timing script with warmup, cuda.synchronize
2. **nsys_profile** (5pt) — Profile with Nsight Systems, analyze kernel time breakdowns
3. **mixed_precision_accumulation** (1pt) — FP16/BF16 accumulation accuracy demo
4. **benchmarking_mixed_precision** (2pt) — Autocast with BF16, benchmark all model sizes
5. **memory_profiling** (4pt) — PyTorch memory profiler on 2.7B model
6. **pytorch_attention** (2pt) — Benchmark attention at various seq_len x d_head
7. **torch_compile** (2pt) — Benchmark torch.compile on attention and full model

### Part 1.3: FlashAttention-2
8. **flash_forward** (15pt) — FA2 forward: (a) PyTorch tiled, (b) Triton kernel, (c) causal masking
9. **flash_backward** (5pt) — FA2 backward with recomputation via torch.compile
10. **flash_benchmarking** (5pt) — Compare Triton FA2 vs vanilla PyTorch attention

### Part 2: Distributed Data Parallel Training
11. **distributed_communication_single_node** (5pt) — Benchmark all-reduce (Gloo+CPU, NCCL+GPU)
12. **naive_ddp** (5pt) — Naive DDP: all-reduce each param after backward
13. **naive_ddp_benchmarking** (3pt) — Benchmark naive DDP on XL model, 1 node x 2 GPUs
14. **minimal_ddp_flat_benchmarking** (2pt) — Flatten all grads into one all-reduce
15. **ddp_overlap_individual_parameters** (5pt) — DDP with overlapped per-param communication
16. **ddp_overlap_individual_parameters_benchmarking** (1pt) — Benchmark + Nsight comparison
17. **ddp_overlap_bucketed** (8pt) — DDP with bucketed gradient communication
18. **ddp_bucketed_benchmarking** (3pt) — Benchmark bucketed DDP, vary bucket sizes
19. **communication_accounting** (10pt) — 4D parallelism memory/communication analysis (XXL model)

### Part 3: Optimizer State Sharding
20. **optimizer_state_sharding** (15pt) — ZeRO-like optimizer sharding wrapper
21. **optimizer_state_sharding_accounting** (5pt) — Memory profiling and comparison with ZeRO

## Test Commands
```bash
uv run pytest -k test_flash_forward_pass_pytorch
uv run pytest -k test_flash_forward_pass_triton
uv run pytest -k test_flash_backward_pytorch
uv run pytest -k test_flash_backward_triton
uv run pytest tests/test_ddp_individual_parameters.py
uv run pytest tests/test_ddp.py
uv run pytest tests/test_sharded_optimizer.py
```

## Adapter Interface (tests/adapters.py)
All test hooks that need implementation:
- `get_flashattention_autograd_function_pytorch()` — Return FA2 autograd.Function (PyTorch only)
- `get_flashattention_autograd_function_triton()` — Return FA2 autograd.Function (Triton kernel)
- `get_ddp_individual_parameters(module)` — Return DDP wrapper (per-param async all-reduce)
- `ddp_individual_parameters_on_after_backward(ddp_model, optimizer)` — Post-backward sync
- `get_ddp_bucketed(module, bucket_size_mb)` — Return bucketed DDP wrapper
- `ddp_bucketed_on_after_backward(ddp_model, optimizer)` — Post-backward sync for bucketed
- `ddp_bucketed_on_train_batch_start(ddp_model, optimizer)` — Pre-step callback
- `get_sharded_optimizer(params, optimizer_cls, **kwargs)` — Return sharded optimizer

## Key Equations — FlashAttention-2

### Forward (Algorithm 1):
- S = QK^T / sqrt(d), P = softmax(S), O = PV
- Online softmax with running max m and denominator l
- L_i = m_i^(Tk) + log(l_i^(Tk))  (logsumexp stored for backward)

### Backward (Eqs 13-19):
- D = rowsum(dO * O)
- P_ij = exp(S_ij - L_i)  (recomputed, not stored)
- dV += P^T dO, dP = dO V^T
- dS_ij = P_ij * (dP_ij - D_i)
- dQ = dS K / sqrt(d), dK = dS^T Q / sqrt(d)

## Conventions
- All code goes in `cs336_systems/`
- Benchmarking scripts should support CLI args for model size, context length, precision, etc.
- Use `torch.cuda.synchronize()` before timing GPU operations
- Warmup steps before measurement (5 warmup, 10 measurement as default)
- Debug distributed with Gloo+CPU, benchmark with NCCL+GPU
