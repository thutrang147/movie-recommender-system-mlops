# Movie Recommender System (MLOps)

This project uses the MovieLens dataset and is managed with uv.

## 1. Requirements

- macOS or Windows
- Python 3.10.x (required by this project)
- Git
- uv

## 2. Install uv

### macOS

Option A (Homebrew):

```bash
brew install uv
```

Option B (official installer):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 3. Clone The Repository

```bash
git clone <your-repo-url>
cd movie-recommender-system-mlops
```

## 4. Create Environment And Install Dependencies

Run this in the project root:

```bash
uv sync --dev
```

## 5. Activate Virtual Environment

### macOS

```bash
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run activation again.

### Windows (CMD)

```bat
.venv\Scripts\activate.bat
```

## 6. Data Prerequisite

By default, data loading expects these files in data/raw:

- ratings.dat
- movies.dat
- users.dat

If they are missing, loading will fail with FileNotFoundError.

## 7. Run The Project Steps (UV First)

### 7.1 Load and clean data

```bash
uv run python src/data/load_data.py
```

This writes cleaned CSV files into data/interim by default.

### 7.2 Validate data quality and save report

```bash
uv run python src/data/validate_data.py --save-report
```

This saves a report to docs/data_quality_report.md.

### 7.3 Build processed parquet artifacts

```bash
uv run python src/data/ingest.py
```

This writes parquet files to data/processed.

Use `--from-raw` to build directly from `data/raw/*.dat`:

```bash
uv run python src/data/ingest.py --from-raw
```

### 7.4 Run tests

```bash
uv run pytest
```

## 8. DVC Workflow

Pull tracked processed artifacts:

```bash
uv run python -m dvc pull
```

Rebuild and re-track processed artifacts:

```bash
uv run python src/data/ingest.py
uv run python -m dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet
uv run python -m dvc push
```

The configured default remote is in `.dvc/config`.

## 9. Optional: Run Notebook

```bash
uv run jupyter lab
```

Then open notebooks/01_eda.ipynb.

## 10. Useful Custom Paths

Load data with custom folders:

```bash
uv run python src/data/load_data.py --raw-dir <raw_data_dir> --output-dir <interim_output_dir>
```

Validate with custom folders/report path:

```bash
uv run python src/data/validate_data.py --raw-dir <raw_data_dir> --interim-dir <interim_dir> --report-path <report_file.md> --save-report
```

Build parquet with custom paths:

```bash
uv run python src/data/ingest.py --interim-dir <interim_dir> --output-dir <processed_dir>
```

## 11. Makefile Shortcuts

```bash
make install
make data-csv
make validate
make report
make ingest-parquet
make dvc-pull
```

## 12. Dependency Management

Add a runtime dependency:

```bash
uv add <package>
```

Add a development dependency:

```bash
uv add --dev <package>
```

After dependency changes, commit both pyproject.toml and uv.lock.

## 13. Troubleshooting

- Wrong Python version:

```bash
uv python install 3.10
uv sync --python 3.10 --dev
```

- Verify Python in the environment:

```bash
uv run python --version
```

- DVC command not found in shell:

```bash
uv run python -m dvc version
```
