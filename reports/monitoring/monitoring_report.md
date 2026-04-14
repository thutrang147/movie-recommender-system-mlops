# Monitoring Report

## Service Metrics
- request_count: 46
- success_count: 41
- error_count: 5
- success_rate: 0.8913
- error_rate: 0.1087
- avg_latency_ms: 13.68
- p95_latency_ms: 57.49

## Data Behavior
- unique_users: 12
- personalized_count: 30
- personalized_rate: 0.6522
- fallback_count: 16
- fallback_rate: 0.3478
- unknown_user_rate: 0.3478
- requests_per_user: 3.8333
- avg_top_k: 4.09

## Drift
- drift_score: 0.5483

### Top Item Shift (Production Top 10)
- item_id=13: train_prob=0.0001, prod_prob=0.1479
- item_id=12: train_prob=0.0001, prod_prob=0.1479
- item_id=11: train_prob=0.0011, prod_prob=0.0592
- item_id=2858: train_prob=0.0043, prod_prob=0.0414
- item_id=593: train_prob=0.0031, prod_prob=0.0414
- item_id=1196: train_prob=0.0037, prod_prob=0.0414
- item_id=260: train_prob=0.0036, prod_prob=0.0414
- item_id=527: train_prob=0.0028, prod_prob=0.0355
- item_id=589: train_prob=0.0032, prod_prob=0.0296
- item_id=2028: train_prob=0.0033, prod_prob=0.0296

## Alerts
- Alert: error_rate=0.1087 exceeded threshold 0.02.
- Drift warning: recommendation distribution drift_score=0.5483 exceeds threshold.
