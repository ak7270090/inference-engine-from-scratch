# Harness report: http://localhost:8000

Generated 2026-10-18 16:31 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 23.9 -> 363.2 tok/s (15.2x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 363.9 tok/s | error rate <= 0% | PASS | E2E p99 22505 ms |
| prefix_reuse | shared_prefix | TTFT p50 138 -> 45 ms (67% lower) | >= 50% lower | PASS | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 48 -> 111 ms (2.3x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 45 ms, ITL p50 41.7 ms, 23.8 tok/s | no errors | PASS | 4 requests at 1 user |
