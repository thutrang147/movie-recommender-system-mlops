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
- recall_at_10: 0.0270
- map_at_10: 0.0219
- coverage: 0.2841

## Personalized Hybrid (SVD + Popularity)
- best_alpha: 0.50
- users_evaluated: 5968
- recall_at_10: 0.0488
- map_at_10: 0.0436
- coverage: 0.0647

## Notes
Popularity baseline uses most_popular_items.parquet; personalized SVD uses svd_model.pkl.
