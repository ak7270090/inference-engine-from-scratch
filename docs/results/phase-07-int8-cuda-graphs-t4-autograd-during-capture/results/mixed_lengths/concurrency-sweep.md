# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 19:04 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 16.9 | 16.9 | 62 | 80 | 58.6 | 68.1 | 16849 | 29360 |
| 4 | 64 | 0 | 64.9 | 16.6 | 179 | 223 | 59.1 | 114.4 | 17634 | 29800 |
| 16 | 64 | 32 | 114.2 | 15.8 | 180 | 206 | 61.8 | 117.1 | 9547 | 30794 |
| 64 | 64 | 64 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
