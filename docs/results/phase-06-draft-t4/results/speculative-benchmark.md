# Speculative decoding: draft, k=4, target Qwen/Qwen3-1.7B

| Prompt | Baseline tok/s | Speculative tok/s | Speed-up | Acceptance | Tokens per pass |
|---|---|---|---|---|---|
| code_edit | 27.6 | 24.2 | 0.88x | 84% | 4.13 |
| chat | 27.7 | 17.5 | 0.63x | 54% | 3.05 |
