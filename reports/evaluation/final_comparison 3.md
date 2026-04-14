# Final Model Comparison

| Model | users_evaluated | Recall@10 | MAP@10 | HitRate@10 | Coverage |
|---|---:|---:|---:|---:|---:|
| Baseline | 5968 | 0.0443 | 0.0380 | 0.3750 | 0.0316 |
| SVD | 5968 | 0.0270 | 0.0219 | 0.2792 | 0.2841 |
| BPR | 5968 | 0.0601 | 0.0460 | 0.4403 | 0.2583 |
| Content-based | 5968 | 0.0131 | 0.0064 | 0.1072 | 0.4139 |

## Notes
- All models evaluated on the same test split with top_k=10 and relevance_threshold=4.0.
- Baseline row uses the most_popular strategy from baseline_report.md.
