# Movie Recommender System (MLOps)

This project uses the MovieLens dataset and is managed with uv.

## Quick Start

Choose one workflow below.

### Option A - Use shared processed artifacts (DVC)

```bash
uv sync
uv run dvc pull
uv run pytest
```

### Option B - Build processed artifacts locally from raw data

```bash
uv sync
uv run python src/data/load_data.py
uv run python src/data/validate_data.py --save-report
uv run python src/data/ingest.py
uv run pytest
```

Use Option A when you only need team-shared processed artifacts.
Use Option B when you have raw data locally and want to rebuild artifacts.

## Data Flow (Raw -> Interim -> Processed)

```text
data/raw/*.dat
  -> load_data.py -> data/interim/*_cleaned.csv
  -> validate_data.py -> docs/data_quality_report.md (optional)
  -> ingest.py -> data/processed/*.parquet
  -> dvc add/push/pull -> DVC remote
```

## Data Scripts Overview

- `src/data/load_data.py`: raw DAT -> cleaned CSV in `data/interim`.
- `src/data/validate_data.py`: quality checks + optional markdown report.
- `src/data/ingest.py`: cleaned CSV (or `--from-raw`) -> parquet in `data/processed`.

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

Option B (Official Installer):

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

Run the following command from the project root:

```bash
uv sync
```

## 5. Activate Virtual Environment

This step is optional when using `uv run ...` commands.

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

By default, the data-loading workflow expects the following files in data/raw:

- ratings.dat
- movies.dat
- users.dat

If any file is missing, execution fails with FileNotFoundError.

## 7. Run Project Workflow (UV)

### 7.1 Load and Clean Data

```bash
uv run python src/data/load_data.py
```

This writes cleaned CSV files into data/interim by default.

### 7.2 Validate Data Quality and Save Report

```bash
uv run python src/data/validate_data.py --save-report
```

This saves a report to docs/data_quality_report.md.

### 7.3 Build Processed Parquet Artifacts

```bash
uv run python src/data/ingest.py
```

This writes parquet files to data/processed.

Use `--from-raw` to build directly from `data/raw/*.dat`:

```bash
uv run python src/data/ingest.py --from-raw
```

### 7.4 Run Tests

```bash
uv run pytest
```

## 8. DVC Workflow

Use DVC when you want to download or share tracked processed artifacts.

Pull shared artifacts from the default remote:

```bash
uv run dvc pull
```

After rebuilding processed artifacts locally, update tracked outputs and push them:

```bash
uv run python src/data/ingest.py
uv run dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet
git add .
git commit -m "Update processed parquet artifacts"
git push
uv run dvc push
```

Note:

- `git push` shares code and DVC metadata.
- `dvc push` shares the tracked artifact files to the DVC remote.

The configured default remote is defined in `.dvc/config`.

You can inspect active DVC settings with:

```bash
uv run dvc config --list
```

### Cloud DVC Setup (Team)

Use this only when pulling/pushing data from the shared Google Drive remote.

1. Request access to the shared drive from the project maintainer.
2. Configure local OAuth values (do not commit these values):

```bash
uv run dvc remote modify --local gdrive_remote gdrive_client_id "<client_id>"
uv run dvc remote modify --local gdrive_remote gdrive_client_secret "<client_secret>"
```

3. Authenticate and pull:

```bash
uv run dvc pull
```

If local data is enough for your task, you can skip cloud pull and run:

```bash
uv run python src/data/ingest.py
```

## 9. Optional: Run Notebook

```bash
uv run jupyter lab
```

Open notebooks/01_eda.ipynb after Jupyter Lab starts.

## 10. Custom Path Options

Load data with custom directories:

```bash
uv run python src/data/load_data.py --raw-dir <raw_data_dir> --output-dir <interim_output_dir>
```

Validate data with custom directories and report path:

```bash
uv run python src/data/validate_data.py --raw-dir <raw_data_dir> --interim-dir <interim_dir> --report-path <report_file.md> --save-report
```

Build parquet artifacts with custom directories:

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

After dependency updates, commit both pyproject.toml and uv.lock.

## 13. Troubleshooting

- Python version mismatch:

```bash
uv python install 3.10
uv sync --python 3.10
```

- Verify Python in the environment:

```bash
uv run python --version
```

- DVC command not found in shell:

```bash
uv run dvc version
```

- DVC import/config errors after dependency changes:

```bash
uv sync
uv run dvc version
```
