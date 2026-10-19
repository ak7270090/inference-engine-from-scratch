# Harness report: http://localhost:8000

Generated 2026-10-18 11:23 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 0.0 -> 0.0 tok/s (0.0x) | >= 4.0x at 64 users | FAIL | levels [1, 4, 16, 64] |
| capacity | burst | 64/64 errors, 0.0 tok/s | error rate <= 0% | FAIL | E2E p99 0 ms |
| prefix_reuse | shared_prefix | TTFT p50 0 -> 0 ms (100% lower) | >= 50% lower | FAIL | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 0 -> 0 ms (0.0x) | <= 2.0x | FAIL | 16 long prompts arriving |
| streaming | short_chat | 0 chunks, 0 tokens | 128 tokens, [DONE] | FAIL | ClientPayloadError: Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'> |
| latency | short_chat | TTFT p50 0 ms, ITL p50 0.0 ms, 0.0 tok/s | no errors | FAIL | 4 requests at 1 user |
