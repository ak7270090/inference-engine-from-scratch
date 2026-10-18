# Step time vs sequence length (cuda-float16, no KV cache)

| Sequence length | Median step (ms) | Min step (ms) |
|---|---|---|
| 16 | 35.5 | 35.2 |
| 64 | 32.6 | 32.3 |
| 128 | 33.8 | 33.0 |
| 256 | 34.2 | 33.3 |
| 512 | 48.2 | 47.5 |
| 1024 | 102.2 | 101.4 |
