# Harness report: http://localhost:8000

Generated 2026-10-18 11:45 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 23.3 -> 599.3 tok/s (25.7x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 606.7 tok/s | error rate <= 0% | PASS | E2E p99 13494 ms |
| prefix_reuse | shared_prefix | TTFT p50 141 -> 46 ms (67% lower) | >= 50% lower | PASS | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 49 -> 111 ms (2.3x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 46 ms, ITL p50 42.3 ms, 23.4 tok/s | no errors | PASS | 4 requests at 1 user |
