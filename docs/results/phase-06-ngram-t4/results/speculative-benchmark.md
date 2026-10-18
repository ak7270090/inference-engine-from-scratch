# Speculative decoding: ngram, k=4, target Qwen/Qwen3-0.6B

| Prompt | Baseline tok/s | Speculative tok/s | Speed-up | Acceptance | Tokens per pass |
|---|---|---|---|---|---|
| code_edit | 28.3 | 86.4 | 3.05x | 57% | 3.20 |
| chat | 28.1 | 51.1 | 1.82x | 44% | 1.86 |
