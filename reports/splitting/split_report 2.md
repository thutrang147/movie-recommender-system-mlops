# Split Report

## Method
- split_method: per-user temporal holdout
- time_column: timestamp_dt
- val_ratio: 0.1
- test_ratio: 0.2
- min_user_interactions: 3
- Why: temporal splitting reduces leakage for recommender evaluation.

## Dataset Size
- total rows: 999611
- train rows: 699826
- val rows: 99895
- test rows: 199890
- eligible users: 6040
- low-history users kept in train: 0

## Notes
Users with fewer than the minimum interaction threshold are kept entirely in train to avoid empty validation/test slices.
