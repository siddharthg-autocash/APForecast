# src/apforecast/core/constants.py

# Paths
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
CONFIG_DIR = f"{DATA_DIR}/config"
REPORTS_DIR = "reports"

MASTER_LEDGER_PATH = f"{PROCESSED_DIR}/master_ledger.parquet"
CONFIG_FILE_PATH = f"{CONFIG_DIR}/vendor_strategy_overrides.xlsx"

# --- COLUMN MAPPING (CRITICAL) ---
# Format: "Your_File_Column_Name": "System_Name"
# UPDATE THE LEFT SIDE to match your CSV headers exactly!
# src/apforecast/core/constants.py

COLUMN_MAP = {
    # "YOUR FILE HEADER"   : "SYSTEM INTERNAL NAME" (Do not change right side!)
    "Check #"              : "Check_ID",
    "Reference"            : "Vendor_ID",
    "Amount"               : "Amount",
    "Post Date"            : "Post_Date",  # Assuming your file header IS "Post_Date"
    "Cleared Date"           : "Clear_Date"  # Assuming your file header IS "Clear_Date"
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