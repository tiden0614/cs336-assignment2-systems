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