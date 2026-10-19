# Harness report: http://localhost:8000

Generated 2026-10-19 12:58 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 68.6 -> 623.3 tok/s (9.1x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 622.4 tok/s | error rate <= 0% | PASS | E2E p99 13153 ms |
| prefix_reuse | shared_prefix | TTFT p50 137 -> 46 ms (66% lower) | >= 50% lower | PASS | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 18 -> 91 ms (5.1x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 45 ms, ITL p50 14.2 ms, 70.2 tok/s | no errors | PASS | 4 requests at 1 user |
