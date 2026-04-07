# Evaluation Report

## Config
- top_k: 10
- relevance_threshold: 4.0

## Popularity Baseline
- users_evaluated: 5968
- recall_at_10: 0.0443
- map_at_10: 0.0380
- coverage: 0.0296

## Personalized SVD
- users_evaluated: 5968
- recall_at_10: 0.0131
- map_at_10: 0.0064
- coverage: 0.4139

## Notes
Popularity baseline uses most_popular_items.parquet; personalized bundle supports both SVD and BPR artifacts.
