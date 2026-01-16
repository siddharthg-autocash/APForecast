# src/apforecast/core/config_loader.py

from datetime import timedelta
import pandas as pd


# ---------------------------------
# Override Rule Builders
# ---------------------------------

def _fixed_lag_rule(lag_days: int):
    def rule(current_age, run_date):
        return 1.0 if current_age >= lag_days else 0.0
    return rule


def _weekday_rule(target_weekday: str, probability: float = 0.6):
    weekday_map = {
        "MONDAY": 0,
        "TUESDAY": 1,
        "WEDNESDAY": 2,
        "THURSDAY": 3,
        "FRIDAY": 4,
    }

    target = weekday_map[target_weekday.upper()]

    def rule(current_age, run_date):
        return probability if run_date.weekday() == target else 0.0

    return rule


# ---------------------------------
# Public Loader
# ---------------------------------
# src/apforecast/core/config_loader.py

from pathlib import Path
import pandas as pd


def load_vendor_overrides(xlsx_path: Path):
    """
    Load vendor override rules.
    If file does not exist or is empty, return empty dict.
    """

    # no overrides configured
    if not xlsx_path.exists() or xlsx_path.stat().st_size == 0:
        return {}

    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception:
        # invalid or unreadable Excel → treat as no overrides
        return {}

    overrides = {}

    for _, row in df.iterrows():
        vendor_id = row.get("Vendor_ID")
        strategy = str(row.get("Strategy", "")).upper()
        param = row.get("Param_1")

        if not vendor_id or strategy == "DEFAULT":
            continue

        if strategy == "FIXED_LAG":
            lag = int(param)
            overrides[vendor_id] = lambda age, run_date, lag=lag: (
                1.0 if age >= lag else 0.0
            )

        elif strategy == "WEEKDAY":
            weekday_map = {
                "MONDAY": 0,
                "TUESDAY": 1,
                "WEDNESDAY": 2,
                "THURSDAY": 3,
                "FRIDAY": 4,
            }
            target = weekday_map.get(str(param).upper())
            if target is not None:
                overrides[vendor_id] = (
                    lambda age, run_date, t=target: 0.6
                    if run_date.weekday() == t else 0.0
                )

    return overrides
