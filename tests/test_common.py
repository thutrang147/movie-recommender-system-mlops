import pandas as pd
import pytest
from pathlib import Path

from src.data.common import (
    canonicalize_dataframe_columns,
    load_table,
    standardize_column_name,
    standardize_dataframe_columns,
    validate_input_files,
)


def test_standardize_column_name():
    assert standardize_column_name("MovieID") == "movieid"
    assert standardize_column_name("User-ID") == "user_id"
    assert standardize_column_name("  ZIP Code  ") == "zip_code"
    assert standardize_column_name("Gender!!!") == "gender"


def test_standardize_dataframe_columns():
    df = pd.DataFrame(columns=["UserID", "Movie ID"])
    result_df = standardize_dataframe_columns(df)
    assert list(result_df.columns) == ["userid", "movie_id"]


def test_canonicalize_dataframe_columns():
    df = pd.DataFrame(columns=["UserID", "MovieID", "Zip-code"])
    result_df = canonicalize_dataframe_columns(df)

    expected = ["user_id", "movie_id", "zip_code"]
    assert list(result_df.columns) == expected


def test_handle_duplicate_columns():
    df = pd.DataFrame(columns=["UserID", "userid"])
    result_df = canonicalize_dataframe_columns(df)

    assert list(result_df.columns) == ["user_id"]


def test_validate_input_files_success(tmp_path: Path):
    for file_name in ("ratings.dat", "movies.dat", "users.dat"):
        (tmp_path / file_name).write_text("sample", encoding="utf-8")

    result = validate_input_files(tmp_path)

    assert result["ratings"] == tmp_path / "ratings.dat"
    assert result["movies"] == tmp_path / "movies.dat"
    assert result["users"] == tmp_path / "users.dat"


def test_validate_input_files_missing(tmp_path: Path):
    (tmp_path / "ratings.dat").write_text("sample", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_input_files(tmp_path)

    message = str(exc_info.value)
    assert "movies.dat" in message
    assert "users.dat" in message


def test_load_table_reads_movie_lens_dat(tmp_path: Path):
    data_file = tmp_path / "ratings.dat"
    data_file.write_text("1::10::4::978300760\n2::20::5::978302109\n", encoding="latin-1")

    result = load_table(data_file, ["UserID", "MovieID", "Rating", "Timestamp"])

    assert list(result.columns) == ["UserID", "MovieID", "Rating", "Timestamp"]
    assert result.shape == (2, 4)
    assert int(result.iloc[0]["UserID"]) == 1