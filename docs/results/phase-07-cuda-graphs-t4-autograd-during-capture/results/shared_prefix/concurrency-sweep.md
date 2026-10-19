# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 18:30 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 24.7 | 24.7 | 45 | 53 | 40.1 | 46.4 | 2595 | 2748 |
| 4 | 64 | 39 | 79.3 | 24.2 | 125 | 130 | 41.0 | 52.1 | 2726 | 2761 |
| 16 | 64 | 64 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
| 64 | 64 | 64 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
