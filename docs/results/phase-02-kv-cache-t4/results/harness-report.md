# Harness report: http://localhost:8000

Generated 2026-10-18 09:29 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 27.5 -> 27.6 tok/s (1.0x) | >= 2.0x at 4 users | XFAIL | levels [1, 4] |
| capacity | burst | 0/64 errors, 27.5 tok/s | error rate <= 0% | PASS | E2E p99 294428 ms |
| prefix_reuse | shared_prefix | TTFT p50 50 -> 50 ms (0% lower) | >= 50% lower | XFAIL | 512 shared of 544 tokens |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 39 ms, ITL p50 36.0 ms, 27.8 tok/s | no errors | PASS | 4 requests at 1 user |
