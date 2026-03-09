## Problem 1: benchmarking_script
### (a)

   model    ctx     mode       mean        std        min        p50        p95        p99        max
-----------------------------------------------------------------------------------------------------
   small    128  forward      18.43       0.85      17.61      18.39      19.21      19.28      19.30
   small    128     both      28.45       0.04      28.42      28.42      28.49      28.50      28.50
   small    256  forward      17.96       0.30      17.78      17.79      18.25      18.29      18.30
   small    256     both      34.03       0.03      33.99      34.04      34.06      34.06      34.06
   small    512  forward      17.83       0.04      17.79      17.83      17.86      17.86      17.86
   small    512     both      51.50       0.01      51.49      51.50      51.51      51.52      51.52
   small   1024  forward      49.34       0.07      49.27      49.35      49.40      49.40      49.40
   small   1024     both     144.26       0.09     144.20     144.22     144.35     144.36     144.37
  medium    128  forward      36.83       2.17      34.72      36.71      38.82      39.01      39.06
  medium    128     both      78.63      25.89      63.49      63.87     104.06     107.63     108.53
  medium    256  forward      37.67       0.46      37.33      37.48      38.12      38.18      38.20
  medium    256     both      80.37       0.09      80.27      80.41      80.44      80.44      80.44
  medium    512  forward      50.45       0.17      50.31      50.40      50.61      50.63      50.63
  medium    512     both     152.91       0.08     152.83     152.94     152.97     152.97     152.97
  medium   1024  forward     138.19       0.10     138.08     138.21     138.27     138.27     138.28
  medium   1024     both     412.77       0.05     412.72     412.76     412.81     412.81     412.81
   large    128  forward      54.48       1.08      53.68      54.04      55.54      55.67      55.70
   large    128     both     110.60       0.20     110.40     110.61     110.78     110.80     110.80
   large    256  forward      56.49       0.43      55.99      56.70      56.76      56.77      56.77
   large    256     both     154.75       0.08     154.70     154.71     154.83     154.84     154.84
   large    512  forward     115.26       0.11     115.16     115.22     115.36     115.38     115.38
   large    512     both     336.07       0.17     335.88     336.15     336.18     336.18     336.18
   large   1024  forward        nan        nan        nan        nan        nan        nan        nan
   large   1024     both        nan        nan        nan        nan        nan        nan        nan
      xl    128  forward      65.90       0.82      65.17      65.74      66.68      66.76      66.78
      xl    128     both     181.56       0.12     181.47     181.53     181.68     181.69     181.69
      xl    256  forward     103.91       0.19     103.76     103.84     104.10     104.12     104.13
      xl    256     both     306.88       0.19     306.71     306.84     307.07     307.09     307.09
      xl    512  forward        nan        nan        nan        nan        nan        nan        nan
      xl    512     both        nan        nan        nan        nan        nan        nan        nan
      xl   1024  forward        nan        nan        nan        nan        nan        nan        nan
      xl   1024     both        nan        nan        nan        nan        nan        nan        nan
    2.7B    128  forward      80.73       0.27      80.44      80.78      80.96      80.97      80.98
    2.7B    128     both     248.28       0.12     248.20     248.21     248.40     248.41     248.42
    2.7B    256  forward     154.34       0.00     154.33     154.33     154.34     154.34     154.34
    2.7B    256     both     451.66       0.13     451.56     451.62     451.79     451.81     451.81
    2.7B    512  forward        nan        nan        nan        nan        nan        nan        nan
    2.7B    512     both        nan        nan        nan        nan        nan        nan        nan
    2.7B   1024  forward        nan        nan        nan        nan        nan        nan        nan
    2.7B   1024     both        nan        nan        nan        nan        nan        nan        nan
-----------------------------------------------------------------------------------------------------

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
In the attention weights computation, div, sum, exp, max, where all take considerable amount of time.

### (d)
With only forward:
45.9ms.

With both forward and backward:
Forward pass takes 46ms. Backward pass takes 40ms.

### (e)
The total runtime is dominated not by any particular kernel, but by torch operations.

## Problem 3: mixed_precision_accumulation
After 1000 accumulations, the errors are also accumulated. When s is defined as f32
and accumulating f32s, the delta is smallest as -0.0001. When s is defined as f16 and
accumuating f16, the delta is largest as 0.0469. When s is defined as f32 and accumulating
f16, delta is larger as -0.0021.

## Problem 4: benchmarking_mixed_precision
### (a)
* the model params within the autocast context: fp32
* the output of the first feed-forward layer: fp16
* the output of layer norm: fp16
* the model's predicted logics: fp16
* the loss: fp32
* gradients: fp16
* the first and second momentums: fp32

### (b)
The sum operation (or any other reduction) is sensitive to FP16.

Using BF16 instead of FP16 doesn't necessarily eliminate the problem. BF16 is more resilient
to the "rounding to zero" issue due to the wider range. However, it can still be a problem
if the values are too small for BF16 as well. Values may not round to zero, but they will 
still experience rounding to a different number because the precision for very small numbers
is much more coarsce compared to FP32.

--- EXPECTED ANSWER ---
No, BF16 has the same dynamic range as FP32, so overflow during variance computation is not 
an issue, and the reduced mantissa precision is tolerable.

### (c)

Similar to context window situation, the precision start to make a difference at a large
enough model at a large enough context length.

For small model, it starts to make a difference when ctx=1024.
For medium, it starts to make a difference when ctx-512.
For large, ctx=512
For xl ctx=256
For 2.7B, ctx=128

   model    ctx     mode   cast       mean        std        min        p50        p95        p99        max
------------------------------------------------------------------------------------------------------------
   small    128  forward   fp32      16.17       0.22      16.03      16.06      16.39      16.42      16.42
   small    128  forward   bf16      17.60       0.13      17.52      17.53      17.73      17.75      17.75
   small    128     both   fp32      28.11       0.13      27.98      28.09      28.23      28.24      28.25
   small    128     both   bf16      30.29       0.13      30.16      30.29      30.40      30.41      30.41
   small    256  forward   fp32      16.52       0.06      16.48      16.49      16.58      16.59      16.59
   small    256  forward   bf16      17.68       0.03      17.65      17.69      17.71      17.71      17.71
   small    256     both   fp32      33.83       0.08      33.78      33.80      33.91      33.92      33.93
   small    256     both   bf16      30.75       0.17      30.64      30.65      30.92      30.94      30.94
   small    512  forward   fp32      17.85       0.01      17.84      17.85      17.86      17.86      17.86
   small    512  forward   bf16      17.63       0.04      17.60      17.60      17.67      17.67      17.67
   small    512     both   fp32      51.56       0.03      51.53      51.56      51.58      51.59      51.59
   small    512     both   bf16      40.91       0.10      40.83      40.87      41.01      41.02      41.02
   small   1024  forward   fp32      48.86       0.09      48.80      48.82      48.95      48.96      48.96
   small   1024  forward   bf16      33.11       0.02      33.09      33.12      33.12      33.12      33.12
   small   1024     both   fp32     144.19       0.04     144.14     144.19     144.22     144.23     144.23
   small   1024     both   bf16     100.94       0.05     100.88     100.96     100.98     100.98     100.98
  medium    128  forward   fp32      32.49       0.08      32.40      32.52      32.55      32.55      32.55
  medium    128  forward   bf16      35.71       0.43      35.43      35.49      36.14      36.20      36.21
  medium    128     both   fp32      66.72       0.52      66.19      66.74      67.19      67.23      67.24
  medium    128     both   bf16      60.93       0.43      60.46      61.04      61.27      61.29      61.29
  medium    256  forward   fp32      32.65       0.46      32.32      32.45      33.10      33.16      33.17
  medium    256  forward   bf16      35.81       0.05      35.76      35.82      35.86      35.86      35.86
  medium    256     both   fp32      80.68       0.13      80.58      80.63      80.80      80.82      80.82
  medium    256     both   bf16      66.35       0.15      66.24      66.29      66.50      66.52      66.52
  medium    512  forward   fp32      50.23       0.14      50.13      50.17      50.37      50.38      50.39
  medium    512  forward   bf16      35.89       0.65      35.48      35.54      36.53      36.62      36.64
  medium    512     both   fp32     153.20       0.10     153.14     153.14     153.30     153.32     153.32
  medium    512     both   bf16      94.69       0.10      94.59      94.69      94.79      94.80      94.80
  medium   1024  forward   fp32     137.67       0.19     137.50     137.64     137.85     137.87     137.88
  medium   1024  forward   bf16      86.99       0.04      86.95      87.01      87.02      87.02      87.02
  medium   1024     both   fp32     412.74       0.05     412.69     412.74     412.78     412.79     412.79
  medium   1024     both   bf16     271.11       0.18     270.97     271.06     271.29     271.31     271.31
   large    128  forward   fp32      50.90       2.15      49.19      50.19      53.00      53.25      53.32
   large    128  forward   bf16      53.70       0.36      53.47      53.52      54.06      54.11      54.12
   large    128     both   fp32     111.32       0.61     110.62     111.61     111.72     111.73     111.73
   large    128     both   bf16      94.58       0.12      94.47      94.56      94.69      94.70      94.71
   large    256  forward   fp32      53.51       0.12      53.38      53.54      53.60      53.61      53.61
   large    256  forward   bf16      57.11       2.02      55.78      56.12      59.10      59.37      59.43
   large    256     both   fp32     156.44       1.28     155.15     156.47     157.58     157.68     157.70
   large    256     both   bf16     130.41       0.79     129.54     130.58     131.05     131.09     131.10
   large    512  forward   fp32     115.26       0.13     115.16     115.22     115.39     115.40     115.41
   large    512  forward   bf16      53.83       0.02      53.81      53.83      53.84      53.84      53.84
   large    512     both   fp32     336.07       0.21     335.90     336.01     336.28     336.31     336.31
   large    512     both   bf16     184.47       3.40     182.11     182.94     187.83     188.27     188.37
   large   1024  forward   fp32        nan        nan        nan        nan        nan        nan        nan
   large   1024  forward   bf16        nan        nan        nan        nan        nan        nan        nan
   large   1024     both   fp32        nan        nan        nan        nan        nan        nan        nan
   large   1024     both   bf16        nan        nan        nan        nan        nan        nan        nan
      xl    128  forward   fp32      67.35       0.90      66.57      67.13      68.22      68.31      68.34
      xl    128  forward   bf16      77.88       1.45      76.21      78.60      78.80      78.82      78.82
      xl    128     both   fp32     186.77       0.59     186.13     186.88     187.25     187.29     187.30
      xl    128     both   bf16     147.69       0.16     147.58     147.63     147.85     147.87     147.88
      xl    256  forward   fp32     103.87       0.15     103.78     103.80     104.02     104.04     104.04
      xl    256  forward   bf16      71.34       0.37      70.96      71.35      71.67      71.70      71.71
      xl    256     both   fp32     305.34       0.19     305.13     305.40     305.48     305.49     305.49
      xl    256     both   bf16     184.31       0.52     184.00     184.02     184.82     184.90     184.91
      xl    512  forward   fp32        nan        nan        nan        nan        nan        nan        nan
      xl    512  forward   bf16     113.11       0.03     113.08     113.12     113.13     113.13     113.13
      xl    512     both   fp32        nan        nan        nan        nan        nan        nan        nan
      xl    512     both   bf16     348.40       0.34     348.02     348.53     348.65     348.66     348.66
      xl   1024  forward   fp32        nan        nan        nan        nan        nan        nan        nan
      xl   1024  forward   bf16        nan        nan        nan        nan        nan        nan        nan
      xl   1024     both   fp32        nan        nan        nan        nan        nan        nan        nan
      xl   1024     both   bf16        nan        nan        nan        nan        nan        nan        nan
    2.7B    128  forward   fp32      81.18       0.01      81.17      81.17      81.19      81.19      81.19
    2.7B    128  forward   bf16      48.24       0.61      47.88      47.91      48.84      48.93      48.95
    2.7B    128     both   fp32     246.62       0.16     246.46     246.62     246.77     246.78     246.78
    2.7B    128     both   bf16     151.61       4.28     146.81     153.04     154.80     154.96     154.99
    2.7B    256  forward   fp32     154.27       0.08     154.18     154.31     154.32     154.32     154.32
    2.7B    256  forward   bf16      61.01       0.41      60.70      60.85      61.42      61.47      61.48
    2.7B    256     both   fp32     450.30       0.12     450.21     450.27     450.42     450.43     450.43
    2.7B    256     both   bf16     207.81       0.04     207.76     207.81     207.85     207.85     207.85
    2.7B    512  forward   fp32        nan        nan        nan        nan        nan        nan        nan
    2.7B    512  forward   bf16        nan        nan        nan        nan        nan        nan        nan
    2.7B    512     both   fp32        nan        nan        nan        nan        nan        nan        nan
    2.7B    512     both   bf16        nan        nan        nan        nan        nan        nan        nan
    2.7B   1024  forward   fp32        nan        nan        nan        nan        nan        nan        nan
    2.7B   1024  forward   bf16        nan        nan        nan        nan        nan        nan        nan
    2.7B   1024     both   fp32        nan        nan        nan        nan        nan        nan        nan
    2.7B   1024     both   bf16        nan        nan        nan        nan        nan        nan        nan
------------------------------------------------------------------------------------------------------------

## Problem 5: memory_profiling
### (a)
On the forward path, the memory time looks like a pyramid. Probably allocation of earlier steps are held
in memory for longer (spaning the whole step) and later steps shorter (allocated and released). The backwards
phase also looks like a pyramid, but probably reversed -- later step allocations are held in memory for longer
and earlier steps shorter.

### (b)

|     | Fwd  | Both |
|-----|------|------|
| 128 | 17.9 | 25.4 |
| 256 | 24.2 | 25.5 |
Unit in GB.

### (c)

| fwd+back | fp32 | mixed |
|----------|------|-------|
| 128      | 25.4 | 25.4  |
| 256      | 25.4 | 25.5  |
Unit in GB.

Mixed precision doesn't seem to affect peak memory utilization.

### (d)
I assume "single precision" here means using fp32 only. 

---
When ctx_len=128, allocations sizes are 5MB, 8MB or 20MB.

batch * seq * d_model * fp32 -> 
4 * 128 * 2560 * 4 / 2^20 = 5MB

---
When ctx_len=256, allocations sizes are 10MB, 32MB or 40MB.

batch * seq * d_model * fp32 -> 
4 * 256 * 2560 * 4 / 2^20 = 10MB

### (e)
Ctx=128, the largest allocations in activation are of size 20MB. This probably came
from FFN layer (batch, seq, d_ff) -> 4 * 128 * 10240 * 4 / 2^20 = 20MB

The static allocations are of size 25MB and 100MB. 
d_model*d_model -> 2560*2560*4/2^20 = 25MB
d_ff * d_model -> 2560*10240*4/2^20 = 100MB

So my guess is that these static allocations are weights in various layers that scale
with d_model or d_ff.

## Problem 6: pytorch_attention
### (a)
d_head  seq_len   fwd_mean    fwd_std   bwd_mean    bwd_std    mem_fwd   peak_bwd
---------------------------------------------------------------------------------
    16      256       0.08       0.01       0.09       0.01       10.6       24.9
    16     1024       0.11       0.00       0.21       0.00       50.3      146.8
    16     4096       2.29       0.01       4.46       0.02      536.3     2074.3
    16     8192       8.78       0.01      17.19       0.03     2080.3     8228.3
    16    16384        nan        nan        nan        nan        nan        nan
    32      256       0.08       0.01       0.09       0.02       19.3       25.5
    32     1024       0.12       0.00       0.24       0.05       52.3      149.3
    32     4096       2.34       0.00       4.53       0.02      544.3     2084.3
    32     8192       9.03       0.01      17.43       0.02     2096.3     8248.3
    32    16384        nan        nan        nan        nan        nan        nan
    64      256       0.08       0.00       0.09       0.01       20.3       26.8
    64     1024       0.13       0.00       0.25       0.04       56.3      154.3
    64     4096       2.51       0.01       4.88       0.02      560.3     2104.3
    64     8192       9.70       0.01      18.43       0.05     2128.3     8288.3
    64    16384        nan        nan        nan        nan        nan        nan
   128      256       0.08       0.01       0.09       0.02       22.3       29.3
   128     1024       0.20       0.01       0.39       0.03       64.3      164.3
   128     4096       3.47       0.01       6.63       0.03      592.3     2144.3
   128     8192      13.66       0.04      25.30       0.07     2192.3     8368.3
   128    16384        nan        nan        nan        nan        nan        nan

Analyzing the configuration before the first OOM: d_head=16 seq_len=8192
Q*K -> batch_size * seq_len * seq_len -> 8*8192*8192*4/2^30 = 2GB
W_Q = W_K = W_V = d_head * d_head -> 16*16*4 = 1024 bytes
Q = K = V -> batch_size * seq_len * d_head -> 8*8192*16*4/2^20 = 4MB

Analyzing the first OOM configuration: d_head=16 seq_len=16384.
Q*K -> batch_size * seq_len * seq_len -> 8*16384*16384*4/2^30 = 8GB
W_Q = W_K = W_V = d_head * d_head -> 16*16*4 = 1024 bytes
Q = K = V -> batch_size * seq_len * d_head -> 8*16384*16*4/2^20 = 8MB

Why OOM?
For seq_len=16384, at peak allocation (happens at the backward pass), 
4 * 8GB allocations exist at the same time:
1. the output of the softmax operation, saved by autograd in forward
2. dP = d(softmax), intermediate gradient during backward
3. elementwise operation (ln) of shape (batch, seq, seq), computed in
   the backwards step of softmax. intermediate result
4. dS = output of softmax_backward

## Problem 7: torch_compile
### (a)
     compile d_head  seq_len   fwd_mean    fwd_std   bwd_mean    bwd_std    mem_fwd   peak_bwd
----------------------------------------------------------------------------------------------
        none     16      256       0.07       0.00       0.08       0.01       10.6       24.9
        none     16     8192       8.81       0.03      17.21       0.04     2080.3     8228.3
        none     16    12288      20.57       0.02      39.33       0.06     4648.3    18478.3
        none    128      256       0.07       0.00       0.09       0.01       22.3       29.3
        none    128     8192      13.65       0.04      25.32       0.07     2192.3     8368.3
        none    128    12288      30.98       0.06      57.46       0.22     4816.3    18688.3
     default     16      256       0.11       0.01       0.08       0.00       18.8       24.9
     default     16     8192       8.84       0.02      17.25       0.06     2080.3     8228.3
     default     16    12288      20.63       0.05      39.36       0.05     4648.3    18478.3
     default    128      256       0.14       0.00       0.09       0.03       22.3       29.3
     default    128     8192      13.76       0.05      25.41       0.07     2192.3     8368.3
     default    128    12288      31.20       0.07      57.67       0.16     4816.3    18688.3

The compilation doesn't seem to be effective for either latency or memory util. I think the
reason is that the operation is already mem bandwidth bound and the compilation doesn't help
further; I would expect the compilation to be effective on the actual model code when the load
is small -- hence the latency is dominated by torch scheduling kernels.

### (b)
   model    ctx     mode   cast          compile       mean        std        min        p50        p95        p99        max
-----------------------------------------------------------------------------------------------------------------------------
   small    128  forward   fp32             none      17.21       0.36      16.94      17.08      18.12      18.54      18.68
   small    128  forward   fp32          default      20.22       0.47      19.86      20.05      21.13      21.79      22.91
   small    256  forward   fp32             none      17.87       0.47      17.53      17.68      19.13      19.80      19.83
   small    256  forward   fp32          default      23.04       0.51      22.69      22.86      24.04      25.20      25.48
   small    512  forward   fp32             none      18.54       0.49      18.14      18.30      19.62      20.14      20.18
   small    512  forward   fp32          default      24.13       0.80      23.47      23.73      25.90      26.43      26.69
   small   1024  forward   fp32             none      48.99       0.14      48.79      48.96      49.34      49.43      49.51
   small   1024  forward   fp32          default      37.51       0.12      37.28      37.49      37.76      38.02      38.02
   large    128  forward   fp32             none      52.33      10.09      50.23      50.99      54.61      55.97     151.63
   large    128  forward   fp32          default      67.26      12.71      65.09      65.57      68.81      71.04     192.58
   large    256  forward   fp32             none      54.17       0.30      53.84      54.11      54.41      55.42      55.92
   large    256  forward   fp32          default      68.80      12.78      66.50      66.99      71.33      74.33     194.64
   large    512  forward   fp32             none     115.80       0.19     115.40     115.81     116.05     116.26     116.35
   large    512  forward   fp32          default     100.96       0.20     100.68     100.93     101.46     101.62     101.77
   large   1024  forward   fp32             none        nan        nan        nan        nan        nan        nan        nan
   large   1024  forward   fp32          default        nan        nan        nan        nan        nan        nan        nan
      xl    128  forward   fp32             none      71.63       1.29      70.40      71.19      74.36      76.17      77.06
      xl    128  forward   fp32          default      92.81      15.06      89.52      90.81      95.42     100.83     240.92
      xl    256  forward   fp32             none     104.71       0.17     104.35     104.70     104.97     105.14     105.17
      xl    256  forward   fp32          default      97.57      14.54      93.56      95.98      98.04     102.04     241.21
      xl    512  forward   fp32             none        nan        nan        nan        nan        nan        nan        nan
      xl    512  forward   fp32          default        nan        nan        nan        nan        nan        nan        nan
      xl   1024  forward   fp32             none        nan        nan        nan        nan        nan        nan        nan
      xl   1024  forward   fp32          default        nan        nan        nan        nan        nan        nan        nan

For forward, compilation=default causes latency regression for smaller ctx and latency gain for larger ctx. It also starts causes
high variance for latency on xl model.

   model    ctx     mode   cast          compile       mean        std        min        p50        p95        p99        max
-----------------------------------------------------------------------------------------------------------------------------
   small    128     both   fp32             none      31.47       2.19      28.12      31.25      34.67      36.52      44.99
   small    128     both   fp32          default      28.44       0.80      27.84      28.10      30.34      31.02      31.56
   small    256     both   fp32             none      34.58       0.56      34.15      34.31      35.92      36.23      36.24
   small    256     both   fp32          default      37.69      10.51      35.07      35.99      39.56      42.03     140.76
   small    512     both   fp32             none      52.84       1.08      51.69      52.39      54.81      55.31      55.99
   small    512     both   fp32          default      51.26       1.82      49.12      50.40      54.14      55.32      55.89
   small   1024     both   fp32             none     145.04       0.32     144.40     145.05     145.53     145.67     145.96
   small   1024     both   fp32          default     103.65       9.90     100.89     102.57     103.35     104.61     201.55
   large    128     both   fp32             none     115.10       2.89     111.14     114.38     119.82     122.53     123.86
   large    128     both   fp32          default     120.21      12.36     117.37     118.52     122.53     128.09     241.59
   large    256     both   fp32             none     156.71       0.37     155.84     156.75     157.21     157.47     157.48
   large    256     both   fp32          default     155.02      12.25     152.20     153.34     158.95     161.41     275.17
   large    512     both   fp32             none     341.72       0.65     339.81     341.81     342.53     342.89     342.89
   large    512     both   fp32          default     288.33       8.72     284.75     287.54     288.04     293.43     374.33
   large   1024     both   fp32             none        nan        nan        nan        nan        nan        nan        nan
   large   1024     both   fp32          default        nan        nan        nan        nan        nan        nan        nan
      xl    128     both   fp32             none     183.89       1.95     181.57     183.21     187.63     189.84     190.59
      xl    128     both   fp32          default     225.66      19.08     220.99     222.05     228.76     355.33     359.97
      xl    256     both   fp32             none     312.06       0.75     310.03     312.26     312.96     313.08     313.12
      xl    256     both   fp32          default     308.52      22.30     301.67     303.72     315.40     432.90     436.01
      xl    512     both   fp32             none        nan        nan        nan        nan        nan        nan        nan
      xl    512     both   fp32          default     587.74       8.51     581.42     587.49     588.65     638.12     651.72
      xl   1024     both   fp32             none        nan        nan        nan        nan        nan        nan        nan
      xl   1024     both   fp32          default        nan        nan        nan        nan        nan        nan        nan

The effect of compilation increases now that we add backward pass into the picture. Latency regression overall is reduced for
smaller ctx and latency gain starts at earlier ctx. The compilation is also able to avoid OOM for xl model where ctx=512.