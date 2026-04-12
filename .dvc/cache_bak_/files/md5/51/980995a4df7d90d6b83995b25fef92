# Baseline Recommender Report

## Config
- top_k: 10
- relevance_threshold: 4.0
- selected_strategy: most_popular
- selection_metric: map_at_k

## Validation Metrics
### most_popular
- users_evaluated: 5815
- recall_at_k: 0.0514
- map_at_k: 0.0269
- hit_rate_at_k: 0.2705
- coverage: 0.0316

### weighted_popularity
- users_evaluated: 5815
- recall_at_k: 0.0492
- map_at_k: 0.0255
- hit_rate_at_k: 0.2488
- coverage: 0.0328

## Test Metrics
### most_popular
- users_evaluated: 5968
- recall_at_k: 0.0443
- map_at_k: 0.0380
- hit_rate_at_k: 0.3750
- coverage: 0.0316

### weighted_popularity
- users_evaluated: 5968
- recall_at_k: 0.0458
- map_at_k: 0.0377
- hit_rate_at_k: 0.3681
- coverage: 0.0328

## Notes
Most Popular ranks by item frequency; Weighted Popularity ranks by mean rating multiplied by log1p(count). Selection uses validation MAP@K and final reporting includes both validation and test metrics.
