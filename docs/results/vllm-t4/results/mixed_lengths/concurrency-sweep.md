# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-17 19:13 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 137.4 | 140.5 | 23 | 29 | 7.2 | 8.4 | 2099 | 3711 |
| 4 | 64 | 0 | 412.2 | 105.2 | 29 | 40 | 9.5 | 15.3 | 2737 | 4827 |
| 16 | 64 | 0 | 796.8 | 55.6 | 56 | 63 | 18.0 | 22.1 | 4561 | 9321 |
| 64 | 64 | 0 | 1117.4 | 31.1 | 125 | 155 | 33.1 | 40.0 | 9949 | 15411 |
