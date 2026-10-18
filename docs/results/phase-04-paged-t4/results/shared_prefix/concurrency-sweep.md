# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 08:26 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 26.1 | 26.3 | 51 | 66 | 37.8 | 45.5 | 2444 | 2574 |
| 4 | 64 | 0 | 95.8 | 25.7 | 220 | 234 | 38.5 | 48.7 | 2666 | 2744 |
| 16 | 64 | 0 | 233.0 | 16.5 | 601 | 874 | 55.8 | 116.9 | 4389 | 4418 |
| 64 | 64 | 0 | 248.8 | 4.8 | 1934 | 3405 | 186.2 | 1284.4 | 15116 | 16069 |
