# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 08:19 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 25.1 | 27.9 | 156 | 205 | 35.6 | 41.9 | 1267 | 1324 |
| 4 | 16 | 0 | 78.8 | 27.2 | 484 | 487 | 36.6 | 39.0 | 1624 | 1627 |
| 16 | 16 | 0 | 80.1 | 27.1 | 2865 | 5261 | 36.7 | 43.1 | 4018 | 6389 |
| 64 | 16 | 0 | 79.7 | 27.2 | 2870 | 5280 | 36.6 | 45.2 | 4025 | 6417 |
