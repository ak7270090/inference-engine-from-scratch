# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-19 14:32 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.1 | 27.3 | 67 | 71 | 36.6 | 37.4 | 4716 | 4740 |
| 4 | 64 | 0 | 85.0 | 21.7 | 166 | 170 | 46.1 | 47.2 | 6027 | 6032 |
| 16 | 64 | 0 | 234.9 | 14.9 | 185 | 195 | 65.9 | 77.7 | 8714 | 8746 |
| 64 | 64 | 0 | 432.8 | 6.9 | 487 | 793 | 144.8 | 190.7 | 18777 | 18924 |
