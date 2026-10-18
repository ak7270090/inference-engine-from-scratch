# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 08:41 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.5 | 27.7 | 88 | 92 | 35.9 | 42.1 | 10569 | 18000 |
| 4 | 64 | 0 | 56.3 | 27.1 | 10029 | 18164 | 36.6 | 42.6 | 17909 | 33528 |
| 16 | 64 | 0 | 201.5 | 26.3 | 6908 | 18838 | 37.6 | 45.3 | 17672 | 31728 |
| 64 | 64 | 0 | 231.5 | 26.0 | 18747 | 56108 | 38.0 | 45.6 | 35646 | 73843 |
