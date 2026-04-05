# Personalized KNN Training Report

## Config
- k: 50
- min_k: 1
- sim_name: cosine
- top_k: 10
- relevance_threshold: 4.0

## Tuning
- candidates_evaluated: 5
- selection_metric: map_at_k

## Validation Metrics
- rmse: 1.0272
- users_evaluated: 5813
- recall_at_k: 0.0014
- map_at_k: 0.0006
- hit_rate_at_k: 0.0256
- coverage: 0.3109

## Notes
Item-based KNN model selected by validation MAP@K over a compact search grid (sampled during tuning, evaluated on full set).
