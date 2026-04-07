# Personalized Model Training Report

## Config
- n_factors: 50
- n_epochs: 30
- lr_all: 0.005
- reg_all: 0.02
- top_k: 10
- relevance_threshold: 4.0

## Tuning
- candidates_evaluated: 17
- selection_metric: map_at_k

## Validation Metrics
- rmse: 0.8858
- users_evaluated: 5813
- recall_at_k: 0.0275
- map_at_k: 0.0142
- hit_rate_at_k: 0.1779
- coverage: 0.2835

## Notes
The model was selected by validation MAP@K over a compact SVD search grid, then refit with the winning hyperparameters.
