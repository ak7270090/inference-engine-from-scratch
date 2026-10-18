# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-17 20:19 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 16.9 | 16.9 | 57 | 65 | 59.2 | 62.1 | 3786 | 3818 |
| 4 | 64 | 0 | 16.9 | 16.9 | 11401 | 11482 | 59.2 | 62.1 | 15124 | 15229 |
