# Monitoring Report

## Service Metrics
- request_count: 2
- success_count: 2
- error_count: 0
- success_rate: 1.0000
- error_rate: 0.0000
- avg_latency_ms: 0.05

## Data Behavior
- fallback_rate: 0.5000
- unknown_user_rate: 0.5000
- requests_per_user: 1.0000

## Drift
- drift_score: 0.6887

### Top Item Shift (Production Top 10)
- item_id=13: train_prob=0.0001, prod_prob=0.4000
- item_id=12: train_prob=0.0001, prod_prob=0.4000
- item_id=11: train_prob=0.0011, prod_prob=0.2000

## Alerts
- Alert: fallback_rate=0.5000 exceeded threshold 0.40.
- Drift warning: recommendation distribution drift_score=0.6887 exceeds threshold.
- Drift warning: unknown_user_rate=0.5000 exceeds threshold.
