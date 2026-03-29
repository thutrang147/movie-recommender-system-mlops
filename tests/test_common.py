import pytest
import pandas as pd
from src.data.common import (
    standardize_column_name, 
    standardize_dataframe_columns, 
    canonicalize_dataframe_columns
)

# 1. Test single column name cleaning logic
def test_standardize_column_name():
    assert standardize_column_name("MovieID") == "movieid"
    assert standardize_column_name("User-ID") == "user_id"
    assert standardize_column_name("  ZIP Code  ") == "zip_code"
    assert standardize_column_name("Gender!!!") == "gender"

# 2. Test standardization for an entire DataFrame
def test_standardize_dataframe_columns():
    df = pd.DataFrame(columns=["UserID", "Movie ID"])
    result_df = standardize_dataframe_columns(df)
    assert list(result_df.columns) == ["userid", "movie_id"]

# 3. Test core canonicalization (Mapping to project-specific names)
def test_canonicalize_dataframe_columns():
    # Simulate raw input columns
    df = pd.DataFrame(columns=["UserID", "MovieID", "Zip-code"])
    result_df = canonicalize_dataframe_columns(df)
    
    # Expected output based on COLUMN_ALIASES in common.py
    expected = ["user_id", "movie_id", "zip_code"]
    assert list(result_df.columns) == expected

# 4. Test handling of duplicate columns after normalization
def test_handle_duplicate_columns():
    # If input has both 'UserID' and 'userid', both map to 'user_id'
    df = pd.DataFrame(columns=["UserID", "userid"])
    result_df = canonicalize_dataframe_columns(df)
    
    # The function should drop duplicates, leaving only one 'user_id'
    assert list(result_df.columns) == ["user_id"]