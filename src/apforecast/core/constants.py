# src/apforecast/core/constants.py

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Paths
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
CONFIG_DIR = f"{DATA_DIR}/config"
REPORTS_DIR = "reports"

MASTER_LEDGER_PATH = f"{PROCESSED_DIR}/master_ledger.parquet"
CONFIG_FILE_PATH = f"{CONFIG_DIR}/vendor_strategy_overrides.xlsx"

# --- COLUMN MAPPING (CRITICAL) ---
COLUMN_MAP = {
    "Check #"              : "Check_ID",
    "Reference"            : "Vendor_ID",
    "Amount"               : "Amount",
    "Post Date"            : "Post_Date",
    "Cleared Date"         : "Clear_Date"
}

# Internal System Column Names (Do Not Change These)
COL_CHECK_ID = "Check_ID"
COL_VENDOR_ID = "Vendor_ID"
COL_AMOUNT = "Amount"
COL_POST_DATE = "Post_Date"
COL_CLEAR_DATE = "Clear_Date"
COL_STATUS = "Status"
COL_DAYS_TO_SETTLE = "Days_to_Settle"

# Statuses
STATUS_OPEN = "OPEN"
STATUS_CLEARED = "CLEARED"
STATUS_VOID = "VOID"

# Cohorts
THRESHOLD_SMALL = 10000
THRESHOLD_LARGE = 50000
COHORT_SMALL = "STABLE_SMALL"
COHORT_MEDIUM = "VOLATILE_MED"
COHORT_LARGE = "LAZY_GIANT"

# Strategies
STRAT_FIXED_LAG = "FIXED_LAG"
STRAT_WEEKDAY = "WEEKDAY"
STRAT_EXACT_DATE = "EXACT_DATE"
STRAT_HOLD = "HOLD"
STRAT_PROB_OVERRIDE = "PROBABILITY_OVERRIDE"
STRAT_DEFAULT = "DEFAULT"

# -----------------------------
# Business-day helpers (US Federal calendar)
# -----------------------------
US_CAL = USFederalHolidayCalendar()
USB = CustomBusinessDay(calendar=US_CAL)

def _holiday_strings_between(start, end):
    """Return list of holiday strings 'YYYY-MM-DD' between two dates (inclusive)."""
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        return []
    try:
        hol = US_CAL.holidays(start=start_ts.date(), end=end_ts.date())
        return [d.strftime("%Y-%m-%d") for d in hol]
    except Exception:
        return []

def is_business_day(date):
    """Return True if `date` is a business day (Mon-Fri and not a US federal holiday)."""
    dt = pd.to_datetime(date).date()
    hols = _holiday_strings_between(dt, dt)
    try:
        return bool(np.is_busday(np.datetime_as_string(np.datetime64(dt)), holidays=hols))
    except Exception:
        # Fallback using CustomBusinessDay
        return pd.Timestamp(dt) in pd.date_range(pd.Timestamp(dt), pd.Timestamp(dt), freq=USB)

def prev_business_day(date):
    """Return the previous business day strictly before `date`."""
    dt = pd.to_datetime(date)
    return (dt - USB).normalize()

def next_business_day(date):
    """Return the next business day strictly after `date`."""
    dt = pd.to_datetime(date)
    return (dt + USB).normalize()

def business_days_between(start_date, end_date):
    """
    Return number of business days between two dates.
    Definition: number of business-day boundaries crossed going from start_date -> end_date.
    If start_date >= end_date -> 0.
    """
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    if s >= e:
        return 0
    hols = _holiday_strings_between(s, e)
    try:
        return int(np.busday_count(np.datetime_as_string(np.datetime64(s)), np.datetime_as_string(np.datetime64(e)), holidays=hols))
    except Exception:
        rng = pd.date_range(start=pd.Timestamp(s), end=pd.Timestamp(e), freq=USB)
        return max(0, len(rng) - 1)
