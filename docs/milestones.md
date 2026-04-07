# Project Milestones

## Roadmap

| Week | Deliverable |
|---|---|
| 7-8 | Problem framing, dataset choice, success metrics |
| 9 | Data ingestion, EDA, DVC/data validation |
| 10 | Baseline model training and evaluation |
| 11 | CI, testing, linting, experiment tracking |
| 12 | API serving, batch inference, model registry |
| 13 | Monitoring, drift detection, alerts |
| 14 | Continuous training, promotion, rollback |

## Week 11 Freeze Checklist

- Main model: BPR (`models/personalized/bpr_model.pkl`)
- Benchmark model: most popular baseline
- Support model: content-based
- Test split: `data/split/test.parquet` (do not tune against this split)

## Notes

- This roadmap is included for grading and presentation context.
- The README stays focused on running and understanding the repo.