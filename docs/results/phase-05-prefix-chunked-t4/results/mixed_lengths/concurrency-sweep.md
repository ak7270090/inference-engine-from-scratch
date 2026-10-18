# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 12:16 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.5 | 23.6 | 46 | 56 | 42.2 | 48.4 | 12389 | 21278 |
| 4 | 64 | 0 | 87.7 | 22.5 | 133 | 154 | 43.7 | 83.5 | 13060 | 22286 |
| 16 | 64 | 0 | 184.7 | 12.8 | 182 | 235 | 82.4 | 130.8 | 17387 | 44430 |
| 64 | 64 | 0 | 341.3 | 9.5 | 284 | 576 | 109.9 | 130.8 | 32533 | 50320 |
