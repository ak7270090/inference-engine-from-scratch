# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 06:09 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.9 | 24.0 | 43 | 50 | 41.5 | 48.1 | 12207 | 20875 |
| 4 | 64 | 0 | 90.5 | 23.1 | 86 | 106 | 42.5 | 82.5 | 12745 | 21467 |
| 16 | 64 | 0 | 264.1 | 19.8 | 92 | 134 | 49.3 | 94.0 | 13608 | 26508 |
| 64 | 64 | 1 | 564.9 | 16.5 | 412 | 416 | 62.7 | 71.4 | 18282 | 29715 |
