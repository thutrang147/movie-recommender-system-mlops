# Data Quality Report

- Generated at (UTC): 2026-04-11 19:26:53
- Data source: cleaned CSV files from data/interim

## Dataset Shapes
- ratings: (1000209, 4)
- movies: (3883, 3)
- users: (6040, 5)

## Missing Values
### ratings
- Total missing cells: 0
- Missing by column: none
### movies
- Total missing cells: 0
- Missing by column: none
### users
- Total missing cells: 0
- Missing by column: none

## Duplicate Records
- ratings exact duplicate rows: 0
- movies exact duplicate rows: 0
- users exact duplicate rows: 0
- ratings duplicate (user_id, movie_id) rows: 0

## User ID Checks (ratings.user_id)
- null: 0
- non-numeric (wrong format): 0
- non-integer (wrong format): 0
- non-positive: 0
- unknown user_id (not found in users table): 0

## Movie ID Checks (ratings.movie_id)
- null: 0
- non-numeric (wrong format): 0
- non-integer (wrong format): 0
- non-positive: 0
- unknown movie_id (not found in movies table): 0

## Rating Checks (ratings.rating)
- null: 0
- non-numeric: 0
- out of range [1, 5]: 0

## Timestamp Checks (ratings.timestamp)
- null: 0
- non-numeric: 0
- non-positive: 0
- unparsable as Unix timestamp: 0
- future timestamps: 0

## Quick Summary
- A value greater than 0 in any check indicates a data quality issue to investigate.