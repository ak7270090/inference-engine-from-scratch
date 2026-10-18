# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 17:05 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 22.8 | 22.8 | 47 | 59 | 43.4 | 51.1 | 12844 | 21788 |
| 4 | 64 | 0 | 85.6 | 22.0 | 135 | 167 | 44.4 | 85.3 | 13540 | 22988 |
| 16 | 64 | 0 | 147.3 | 10.3 | 228 | 377 | 91.9 | 166.9 | 21792 | 56306 |
| 64 | 64 | 0 | 184.9 | 5.2 | 339 | 980 | 173.8 | 319.5 | 65700 | 93188 |
