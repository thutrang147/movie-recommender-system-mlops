# Week 14 Retraining Report

- strategy: schedule
- status: evaluated_keep_current
- dry_run: False
- trigger_reason: Schedule-based retraining selected: run every cycle.
- decision: keep_current
- decision_reason: Keep current model: candidate did not satisfy promotion rule (candidate recall=0.045120, current recall=0.060079, candidate coverage=0.033089, current coverage=0.258294).
- drift_score: 0.5398

## Model Selection
- current_model: bpr (v1)
- current_artifact: models/personalized/bpr_model.pkl
- candidate_model: bpr_candidate (candidate-20260413T054040Z)
- candidate_artifact: models/personalized/candidates/bpr_model_20260413T054040Z.pkl

## Metrics
- current_recall_at_k: 0.060079
- candidate_recall_at_k: 0.045120
- current_coverage: 0.258294
- candidate_coverage: 0.033089
- recall_at_k_delta: -0.014958
- coverage_delta: -0.225205

## Training Data Update
- new_feedback_rows: 0
- updated_train_rows: 699826

## Artifacts
- registry_path: models/registry.json
- candidate_artifact: models/personalized/candidates/bpr_model_20260413T054040Z.pkl
- rollback_alias: models/personalized/active_model_prev.pkl

## Promotion Rule
- Promote when candidate recall_at_k is higher than current recall_at_k.
- If recall_at_k is tied, use coverage as the tie-breaker.

## Rollback Strategy
- If a promoted model is unstable in production or degrades KPI, run `make retrain-rollback`.
- The rollback command restores active model to a real artifact path (never to alias path).
