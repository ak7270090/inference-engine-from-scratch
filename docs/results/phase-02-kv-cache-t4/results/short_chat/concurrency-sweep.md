# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 09:39 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.3 | 27.3 | 39 | 42 | 36.3 | 43.5 | 4684 | 4824 |
| 4 | 64 | 0 | 27.4 | 27.5 | 14033 | 14372 | 36.1 | 43.8 | 18684 | 19127 |
