# Monitoring Report

## Service Metrics
- request_count: 13
- success_count: 13
- error_count: 0
- success_rate: 1.0000
- error_rate: 0.0000
- avg_latency_ms: 0.40

## Data Behavior
- fallback_rate: 0.3077
- unknown_user_rate: 0.3077
- requests_per_user: 3.2500

## Drift
- drift_score: 0.5950

### Top Item Shift (Production Top 10)
- item_id=13: train_prob=0.0001, prod_prob=0.1000
- item_id=12: train_prob=0.0001, prod_prob=0.1000
- item_id=527: train_prob=0.0028, prod_prob=0.1000
- item_id=912: train_prob=0.0019, prod_prob=0.1000
- item_id=593: train_prob=0.0031, prod_prob=0.0833
- item_id=318: train_prob=0.0026, prod_prob=0.0667
- item_id=2396: train_prob=0.0027, prod_prob=0.0667
- item_id=11: train_prob=0.0011, prod_prob=0.0500
- item_id=2858: train_prob=0.0043, prod_prob=0.0333
- item_id=1196: train_prob=0.0037, prod_prob=0.0333

## Alerts
- Drift warning: recommendation distribution drift_score=0.5950 exceeds threshold.
