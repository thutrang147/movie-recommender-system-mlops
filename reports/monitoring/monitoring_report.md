# Monitoring Report

## Service Metrics
- request_count: 28
- success_count: 28
- error_count: 0
- success_rate: 1.0000
- error_rate: 0.0000
- avg_latency_ms: 22.14
- p95_latency_ms: 61.92

## Data Behavior
- unique_users: 12
- personalized_count: 18
- personalized_rate: 0.6429
- fallback_count: 10
- fallback_rate: 0.3571
- unknown_user_rate: 0.3571
- requests_per_user: 2.3333
- avg_top_k: 5.00

## Drift
- drift_score: 0.5398

### Top Item Shift (Production Top 10)
- item_id=13: train_prob=0.0001, prod_prob=0.0857
- item_id=12: train_prob=0.0001, prod_prob=0.0857
- item_id=2858: train_prob=0.0043, prod_prob=0.0500
- item_id=593: train_prob=0.0031, prod_prob=0.0500
- item_id=1196: train_prob=0.0037, prod_prob=0.0500
- item_id=260: train_prob=0.0036, prod_prob=0.0500
- item_id=11: train_prob=0.0011, prod_prob=0.0500
- item_id=527: train_prob=0.0028, prod_prob=0.0429
- item_id=589: train_prob=0.0032, prod_prob=0.0357
- item_id=2028: train_prob=0.0033, prod_prob=0.0357

## Alerts
- Drift warning: recommendation distribution drift_score=0.5398 exceeds threshold.
