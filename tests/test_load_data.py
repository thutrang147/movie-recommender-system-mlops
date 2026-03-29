from src.data.load_data import resolve_data_dirs
from pathlib import Path

def test_resolve_data_dirs_defaults():
    """Test if directories resolve to project defaults when no args are provided."""
    raw, interim = resolve_data_dirs(None, None)
    
    assert isinstance(raw, Path)
    assert isinstance(interim, Path)
    # Check if they point to the right folders
    assert raw.name == "raw"
    assert interim.name == "interim"

def test_resolve_data_dirs_custom():
    """Test if custom directory strings are correctly converted to Path objects."""
    custom_raw = "D:/my_data/raw"
    custom_out = "D:/my_data/output"
    raw, interim = resolve_data_dirs(custom_raw, custom_out)
    
    assert raw == Path(custom_raw)
    assert interim == Path(custom_out)