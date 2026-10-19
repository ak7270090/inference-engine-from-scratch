# Harness report: http://localhost:8000

Generated 2026-10-19 14:25 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 27.4 -> 433.2 tok/s (15.8x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 433.5 tok/s | error rate <= 0% | PASS | E2E p99 18887 ms |
| prefix_reuse | shared_prefix | TTFT p50 209 -> 68 ms (68% lower) | >= 50% lower | PASS | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 41 -> 143 ms (3.5x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 65 ms, ITL p50 36.6 ms, 27.3 tok/s | no errors | PASS | 4 requests at 1 user |
