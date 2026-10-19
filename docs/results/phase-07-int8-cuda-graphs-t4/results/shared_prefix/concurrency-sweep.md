# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-19 14:39 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 21.7 | 21.9 | 68 | 71 | 45.6 | 46.6 | 2943 | 2956 |
| 4 | 64 | 0 | 47.6 | 12.1 | 179 | 190 | 82.5 | 84.6 | 5381 | 5389 |
| 16 | 64 | 0 | 109.1 | 7.0 | 332 | 339 | 143.9 | 151.2 | 9384 | 9398 |
| 64 | 64 | 0 | 131.7 | 2.1 | 1142 | 2323 | 470.6 | 656.6 | 30789 | 31094 |
