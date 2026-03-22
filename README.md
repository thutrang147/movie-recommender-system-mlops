## 🛠️ Environment Setup & Reproducibility

This project uses `Makefile` to automate the MLOps pipeline. 

**For Linux / macOS / WSL Users:**
Simply run the following commands:
`make install` (Install dependencies)
`make data` (Ingest data)
`make dvc-pull` (Pull processed data from Google Drive)

**For Windows Users (PowerShell/CMD):**
Since `make` is not natively supported on Windows, you have 2 options:
1. **Option 1 (Recommended):** Use `Git Bash` or `WSL` to run the `make` commands above.
2. **Option 2:** Run the raw commands directly in your terminal:
   - Install dependencies: `pip install -r requirements.txt`
   - Ingest data: `python src/data/ingest.py`
   - Pull data: `dvc pull`