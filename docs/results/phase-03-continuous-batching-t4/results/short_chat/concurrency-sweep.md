# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 07:24 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 26.0 | 26.1 | 40 | 44 | 38.1 | 45.4 | 4905 | 5180 |
| 4 | 64 | 0 | 102.3 | 25.7 | 45 | 112 | 38.6 | 44.4 | 5010 | 5107 |
| 16 | 64 | 0 | 393.0 | 24.9 | 118 | 137 | 39.8 | 47.0 | 5211 | 5251 |
| 64 | 64 | 0 | 394.0 | 24.7 | 5354 | 15785 | 40.0 | 44.0 | 10506 | 20787 |
