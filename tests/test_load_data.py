from pathlib import Path

import pandas as pd

from src.data.load_data import resolve_data_dirs, save_cleaned_tables


def test_resolve_data_dirs_defaults():
    raw, interim = resolve_data_dirs(None, None)

    assert isinstance(raw, Path)
    assert isinstance(interim, Path)
    assert raw.name == "raw"
    assert interim.name == "interim"
    assert raw.parent.name == "data"
    assert interim.parent.name == "data"


def test_resolve_data_dirs_custom():
    custom_raw = "./my_data/raw"
    custom_out = "./my_data/output"
    raw, interim = resolve_data_dirs(custom_raw, custom_out)

    assert raw == Path(custom_raw)
    assert interim == Path(custom_out)


def test_save_cleaned_tables_writes_expected_files(tmp_path: Path):
    tables = {
        "ratings": pd.DataFrame({"user_id": [1], "movie_id": [2], "rating": [4.0]}),
        "movies": pd.DataFrame({"movie_id": [2], "title": ["Toy Story"], "genres": ["Animation"]}),
        "users": pd.DataFrame({"user_id": [1], "gender": ["F"]}),
    }

    save_cleaned_tables(tables, tmp_path)

    for table_name in tables:
        output_path = tmp_path / f"{table_name}_cleaned.csv"
        assert output_path.exists()

    ratings_readback = pd.read_csv(tmp_path / "ratings_cleaned.csv")
    assert list(ratings_readback.columns) == ["user_id", "movie_id", "rating"]
    assert int(ratings_readback.iloc[0]["user_id"]) == 1