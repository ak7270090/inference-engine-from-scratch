# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 07:35 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 22.0 | 22.1 | 98 | 106 | 44.8 | 54.6 | 12755 | 23396 |
| 4 | 64 | 0 | 46.6 | 22.4 | 12195 | 22142 | 44.2 | 51.4 | 21573 | 40074 |
| 16 | 64 | 0 | 171.6 | 22.5 | 8226 | 22120 | 44.1 | 50.7 | 20815 | 37242 |
| 64 | 64 | 0 | 198.8 | 22.3 | 22236 | 64900 | 44.4 | 54.5 | 38004 | 86680 |
