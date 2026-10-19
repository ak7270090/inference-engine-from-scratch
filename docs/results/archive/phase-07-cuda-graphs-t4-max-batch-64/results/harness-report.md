# Harness report: http://localhost:8000

Generated 2026-10-19 12:29 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 68.9 -> 366.6 tok/s (5.3x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 362.9 tok/s | error rate <= 0% | PASS | E2E p99 22561 ms |
| prefix_reuse | shared_prefix | TTFT p50 142 -> 48 ms (66% lower) | >= 50% lower | PASS | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 18 -> 92 ms (5.1x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 48 ms, ITL p50 14.6 ms, 68.5 tok/s | no errors | PASS | 4 requests at 1 user |
