# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 06:41 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 20.8 | 20.8 | 50 | 53 | 48.0 | 55.1 | 13976 | 23816 |
| 4 | 64 | 0 | 20.7 | 20.7 | 39018 | 59517 | 48.0 | 54.6 | 52655 | 76479 |
