# Preprocessing Report

## Cleaning Rules
- Cast id/rating/timestamp fields to numeric types.
- Drop rows with null, non-positive, or out-of-range values.
- Remove ratings whose user/movie ids are missing from the reference tables.
- Duplicate strategy: latest.
- Keep the latest record when the same user rates the same movie multiple times.

## Optional Frequency Filters
- min_user_interactions: 5
- min_movie_interactions: 5
- Why: reduce extreme sparsity and stabilize matrix factorization for training.
- Impact: fewer cold-start edges, better signal density, but lower catalog coverage and a stronger popularity bias.

## Output Impact
- ratings rows before filter: 1000209
- ratings rows after filter: 1000209
- ratings rows removed by frequency filter: 598
- users removed by frequency filter: 0
- movies removed by frequency filter: 290
- output ratings rows: 999611
- output users: 6040
- output movies: 3416

## Notes
Frequency filtering is optional and removes sparse users/items to reduce extreme sparsity, but it can shrink catalog coverage and bias the dataset toward active users and popular movies.
