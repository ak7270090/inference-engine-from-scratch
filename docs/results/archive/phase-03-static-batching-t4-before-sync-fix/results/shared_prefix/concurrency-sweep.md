# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 07:13 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 22.4 | 22.9 | 110 | 125 | 43.2 | 51.2 | 2856 | 2953 |
| 4 | 64 | 0 | 83.7 | 22.8 | 286 | 290 | 43.7 | 49.9 | 3050 | 3129 |
| 16 | 64 | 0 | 130.3 | 22.6 | 3584 | 6789 | 44.0 | 50.0 | 6384 | 9586 |
| 64 | 64 | 0 | 130.0 | 22.6 | 13108 | 26989 | 44.0 | 52.6 | 15979 | 29754 |
