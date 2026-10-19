# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 18:41 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 17.5 | 17.5 | 60 | 67 | 56.8 | 66.1 | 3660 | 3721 |
| 4 | 64 | 40 | 35.4 | 16.9 | 177 | 189 | 58.3 | 69.0 | 3889 | 3920 |
| 16 | 64 | 64 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
| 64 | 64 | 62 | 23.2 | 17.3 | 1859 | 1859 | 57.8 | 59.5 | 5505 | 5505 |
