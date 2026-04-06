# Week 14 Retraining Report

- strategy: schedule
- status: rollback_completed
- trigger_reason: manual rollback flag provided
- decision: rollback
- decision_reason: Active model switched to rollback snapshot alias.

## Metrics
- current_recall_at_k: 0.000000
- candidate_recall_at_k: 0.000000
- current_coverage: 0.000000
- candidate_coverage: 0.000000

## Artifacts
- registry_path: models/registry.json
- candidate_artifact: N/A
- rollback_alias: models/personalized/active_model_prev.pkl

## Rollback Strategy
- If a promoted model is unstable in production or degrades KPI, run `make retrain-rollback`.
- The rollback command switches active model in registry to the `active_model_prev` snapshot.
