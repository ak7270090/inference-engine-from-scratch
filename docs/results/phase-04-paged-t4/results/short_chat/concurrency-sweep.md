# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 08:22 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 26.7 | 26.8 | 40 | 49 | 37.2 | 43.9 | 4788 | 4903 |
| 4 | 64 | 0 | 103.7 | 26.0 | 41 | 83 | 38.1 | 45.4 | 4933 | 5024 |
| 16 | 64 | 0 | 394.7 | 25.2 | 132 | 139 | 39.5 | 50.1 | 5172 | 5269 |
| 64 | 64 | 0 | 1028.2 | 16.9 | 453 | 456 | 60.6 | 78.4 | 7962 | 7966 |
