# Harness report: http://localhost:8000

Generated 2026-10-17 19:07 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 141.7 -> 1926.8 tok/s (13.6x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 1910.2 tok/s | error rate <= 0% | PASS | E2E p99 4287 ms |
| prefix_reuse | shared_prefix | TTFT p50 86 -> 23 ms (73% lower) | >= 50% lower | PASS | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 8 -> 193 ms (24.1x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 24 ms, ITL p50 6.9 ms, 145.8 tok/s | no errors | PASS | 4 requests at 1 user |
