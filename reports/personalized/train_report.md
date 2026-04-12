# Personalized Model Training Report

## Config
- n_factors: 50
- n_epochs: 20
- lr_all: 0.007
- reg_all: 0.02
- top_k: 10
- relevance_threshold: 4.0

## Tuning
- candidates_evaluated: 145
- selection_metric: map_at_k

## Validation Metrics
- rmse: 0.8825
- users_evaluated: 5813
- recall_at_k: 0.0276
- map_at_k: 0.0142
- hit_rate_at_k: 0.1767
- coverage: 0.2613

## Notes
The model was selected by validation MAP@K over an expanded SVD search grid, then refit on train+val with the winning hyperparameters.
