## Problem 1: benchmarking_script
### (a)
The script is cs336_systems/benchmark.py. It offers both single and sweep mode to 
measure time spent forward/both(fwd+bwd).
### (b)
#### both
On my RTX5090, the reported time for both is:

```bash
uv run python -m cs336_systems.benchmark single --model_size small --context_length 128 --device cuda:0 --num_warmup 5 --num_steps 1000 --mode both
Config: model=small, ctx=128, mode=both, warmup=5, steps=1000, device=cuda:0
Mean: 26.12 ms | Std: 4.58 ms
```

#### forward
The reported time for forward is:

```bash
(cs336-systems) ying-zhang@ying-zhang-B760M-DS3H-DDR4:~/workplace/cs336/cs336-assignment2-systems$ uv run python -m cs336_systems.benchmark single --model_size small --context_length 128 --device cuda:0 --num_warmup 5 --num_steps 1000 --mode forward
Config: model=small, ctx=128, mode=forward, warmup=5, steps=1000, device=cuda:0
Mean: 12.69 ms | Std: 1.41 ms
```

### (c)
Yes, using warm up vs. not makes a big difference in the measured results:

```bash
(cs336-systems) ying-zhang@ying-zhang-B760M-DS3H-DDR4:~/workplace/cs336/cs336-assignment2-systems$ uv run python -m cs336_systems.benchmark single --model_size small --context_length 128 --device cuda:0 --num_warmup 0 --num_steps 1000 --mode both
Config: model=small, ctx=128, mode=both, warmup=0, steps=1000, device=cuda:0
  MEAN: 25.09 ms
   STD: 11.47 ms
   MIN: 24.27 ms
   P50: 24.43 ms
   P95: 25.87 ms
   P99: 30.17 ms
   MAX: 383.49 ms
(cs336-systems) ying-zhang@ying-zhang-B760M-DS3H-DDR4:~/workplace/cs336/cs336-assignment2-systems$ uv run python -m cs336_systems.benchmark single --model_size small --context_length 128 --device cuda:0 --num_warmup 10 --num_steps 1000 --mode both
Config: model=small, ctx=128, mode=both, warmup=10, steps=1000, device=cuda:0
  MEAN: 24.69 ms
   STD: 1.41 ms
   MIN: 24.44 ms
   P50: 24.60 ms
   P95: 24.93 ms
   P99: 25.53 ms
   MAX: 67.52 ms
```

## Problem 2: nsys_profile
### (a)
The time more than doubles. Now instead of 25ms, the mean becomes 64ms. The added profiling 
functions add to the total runtime.

### (b)
On the forward path, self_attention takes the most time. A whole transformer block takes 2.823ms. The self_attention takes 2.349ms.

* RMSNorm: 6
* self_attention.qkv_proj: 3
* self_attention.rope: 9
* self_attention.casual_masking_construct: 2
* self_attention.attention_score: 2
* self_attention.casual_masking: 2
* self_attention.attention_weights_computation: 5
* self_attention.output_computation: 2
* self_attention.output_reshaping: 1
* self_attention.output_project: 1 memset + 1 kernel (44us)
* swiglu: 3 memset + 6 kernels
Total: 2 * RMSNorm + self_attention + swiglu = 45

On the backward pass, attention still takes up the most time, but not as dramatic as in the forward pass. A whole backward pass take
3.103ms, and the attention backward pass takes 1.619ms.

### (c)
Besides aten::bmm/aten::mul, in rms_norm, aten::mean takes 2.432us, aten::pow takes 1.249us
In attention_core_computation, aten::div takes 1.760us. In causal_mask_construction, aten::ge takes 1.472us.

### (d)
With only forward:
45.9ms.

With both forward and backward:
Forward pass takes 46ms. Backward pass takes 40ms.

### (e)
The total runtime is dominated not by any particular kernel, but by torch operations.