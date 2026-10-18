# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 05:53 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.7 | 23.9 | 57 | 73 | 41.6 | 48.9 | 2691 | 2769 |
| 4 | 64 | 0 | 88.0 | 23.4 | 225 | 229 | 42.4 | 48.6 | 2908 | 2953 |
| 16 | 64 | 0 | 222.9 | 15.6 | 536 | 860 | 59.2 | 122.5 | 4593 | 4614 |
| 64 | 64 | 0 | 251.3 | 4.9 | 1918 | 3373 | 182.4 | 1265.9 | 14853 | 15875 |
