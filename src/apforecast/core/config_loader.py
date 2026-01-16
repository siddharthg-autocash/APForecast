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

def load_vendor_overrides(xlsx_path):
    """
    Returns:
        dict[vendor_id -> callable]
    """

    df = pd.read_excel(xlsx_path)

    overrides = {}

    for _, row in df.iterrows():
        vendor_id = row["Vendor_ID"]
        strategy = row["Strategy"].upper()
        param = row.get("Param_1")

        if strategy == "FIXED_LAG":
            overrides[vendor_id] = _fixed_lag_rule(int(param))

        elif strategy == "WEEKDAY":
            overrides[vendor_id] = _weekday_rule(str(param))

        elif strategy == "DEFAULT":
            continue

    return overrides
