# Harness report: http://localhost:8000

Generated 2026-10-18 18:25 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 17.4 -> 0.0 tok/s (0.0x) | >= 4.0x at 64 users | FAIL | levels [1, 4, 16, 64] |
| capacity | burst | 64/64 errors, 0.0 tok/s | error rate <= 0% | FAIL | E2E p99 0 ms |
| prefix_reuse | shared_prefix | TTFT p50 181 -> 61 ms (67% lower) | >= 50% lower | PASS | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 65 -> 138 ms (2.1x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 59 ms, ITL p50 56.6 ms, 17.6 tok/s | no errors | PASS | 4 requests at 1 user |
