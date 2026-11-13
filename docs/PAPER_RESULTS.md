# Paper-reported results

CoSpaDi results from [arXiv:2509.22075v8](https://arxiv.org/html/2509.22075v8),
Table 1 (Llama-3.2-1B) and Table 3 (Llama-3-8B and Qwen3-8B).
Accuracy is the eight-task average (%); lower perplexity is better.

| Model | CR | Avg. accuracy ↑ | WikiText PPL ↓ | LAMBADA PPL ↓ |
| :-- | --: | --: | --: | --: |
| Llama-3.2-1B | 0.2 | 42.7 | 64 | 35 |
| Llama-3.2-1B | 0.3 | 33.7 | 290 | 660 |
| Llama-3.2-1B | 0.4 | 27.5 | 800 | 9200 |
| Llama-3-8B | 0.2 | 61.8 | 20 | 4.3 |
| Llama-3-8B | 0.3 | 53.7 | 45 | 9.2 |
| Llama-3-8B | 0.4 | 40.7 | 180 | 120 |
| Qwen3-8B | 0.2 | 65.7 | 18 | 4.9 |
| Qwen3-8B | 0.3 | 59.97 | 23 | 6.3 |
| Qwen3-8B | 0.4 | 50.3 | 36 | 15 |

Qwen3-0.6B accuracy and WikiText perplexity curves appear in Figure 5.
See the paper for per-task results, baselines, ablations and grouped compression.

Machine-readable table values: [paper_results.csv](paper_results.csv).
