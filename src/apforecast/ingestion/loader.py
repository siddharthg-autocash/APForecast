# src/apforecast/ingestion/loader.py

from pathlib import Path

# Allow either CSV or XLSX for each required input
REQUIRED_INPUTS = {
    "bank_cleared": {".csv", ".xlsx"},
    "issued_checks": {".csv", ".xlsx"},
}


def get_raw_data_dir(base_path: Path, run_date_str: str) -> Path:
    """
    Resolve and validate the raw data directory for a given run date.
    Accepts either CSV or XLSX inputs.
    """

    raw_dir = base_path / "data" / "raw" / run_date_str

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}"
        )

    files = {p.name for p in raw_dir.iterdir() if p.is_file()}

    missing = []

    for base_name, exts in REQUIRED_INPUTS.items():
        if not any(f"{base_name}{ext}" in files for ext in exts):
            missing.append(
                f"{base_name}{list(exts)}"
            )

    if missing:
        raise FileNotFoundError(
            f"Missing required files in {raw_dir}: {missing}"
        )

    return raw_dir
