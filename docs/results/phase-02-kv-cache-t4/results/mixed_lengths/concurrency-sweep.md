# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 10:05 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.9 | 27.9 | 38 | 41 | 35.6 | 42.8 | 10579 | 17959 |
| 4 | 64 | 0 | 28.0 | 28.0 | 28608 | 44007 | 35.4 | 42.4 | 38816 | 56827 |
