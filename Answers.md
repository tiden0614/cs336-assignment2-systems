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
