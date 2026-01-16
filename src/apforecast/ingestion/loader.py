# src/apforecast/ingestion/loader.py

from pathlib import Path


REQUIRED_FILES = {
    "bank_cleared.csv",
    "issued_checks.csv",
}


def get_raw_data_dir(base_path: Path, run_date_str: str) -> Path:
    """
    Resolve and validate the raw data directory for a given run date.
    """

    raw_dir = base_path / "data" / "raw" / run_date_str

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}"
        )

    present_files = {p.name for p in raw_dir.iterdir() if p.is_file()}
    missing = REQUIRED_FILES - present_files

    if missing:
        raise FileNotFoundError(
            f"Missing required files in {raw_dir}: {missing}"
        )

    return raw_dir
