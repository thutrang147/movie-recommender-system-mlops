# BPR Personalized Model Training Report

## Config
- factors: 64
- learning_rate: 0.03
- reg: 0.001
- epochs: 30
- n_samples_per_epoch: 200000
- top_k: 10
- relevance_threshold: 4.0
- patience: 5

## Validation Metrics
- users_evaluated: 5813
- recall_at_k: 0.0742
- map_at_k: 0.0373
- hit_rate_at_k: 0.3406
- coverage: 0.2583

## Early Stopping
- best_epoch: 26
- best_val_map: 0.0373

## Notes
BPR is trained with sampled pairwise triplets and early stopping on validation MAP@K.
