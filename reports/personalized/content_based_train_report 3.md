# Content-Based Personalized Model Training Report

## Config
- top_k: 10
- relevance_threshold: 4.0
- min_df: 2
- max_features: 12000
- ngram_range: (1, 2)

## Validation Metrics
- users_evaluated: 6038
- recall_at_k: 0.0000
- map_at_k: 0.0000
- hit_rate_at_k: 0.0000
- coverage: 0.4212

## Notes
Content-based ranking uses TF-IDF over movie title and genres with user preference profiles.
