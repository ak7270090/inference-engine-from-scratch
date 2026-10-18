# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 07:01 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 21.9 | 21.9 | 48 | 56 | 45.5 | 53.1 | 13236 | 22672 |
| 4 | 64 | 0 | 83.0 | 21.2 | 94 | 122 | 46.5 | 86.0 | 13862 | 23448 |
| 16 | 64 | 0 | 241.9 | 18.1 | 101 | 148 | 53.6 | 102.5 | 14968 | 28751 |
| 64 | 64 | 0 | 265.9 | 18.8 | 14680 | 41989 | 51.4 | 103.8 | 32251 | 64964 |
