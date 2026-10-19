# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 18:44 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 24.8 | 24.8 | 44 | 49 | 40.0 | 46.7 | 11828 | 20120 |
| 4 | 64 | 40 | 28.1 | 24.0 | 51 | 130 | 41.0 | 57.9 | 5509 | 12941 |
| 16 | 64 | 48 | 181.1 | 23.2 | 140 | 142 | 42.8 | 48.1 | 9679 | 21071 |
| 64 | 64 | 62 | 5.3 | 13.5 | 224 | 267 | 68.8 | 106.6 | 1518 | 1657 |
