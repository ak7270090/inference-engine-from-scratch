# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-19 13:14 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 64.1 | 65.7 | 46 | 61 | 14.4 | 22.1 | 4487 | 8228 |
| 4 | 64 | 0 | 122.8 | 31.1 | 108 | 145 | 32.1 | 64.9 | 8616 | 17442 |
| 16 | 64 | 0 | 192.5 | 13.0 | 178 | 229 | 81.0 | 127.7 | 16948 | 43430 |
| 64 | 64 | 0 | 345.6 | 9.7 | 290 | 534 | 107.3 | 128.5 | 31871 | 49860 |
