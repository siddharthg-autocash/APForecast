# src/apforecast/core/constants.py

import pyarrow as pa

# Master Ledger Schema

MASTER_LEDGER_SCHEMA = pa.schema([
    # Identity
    pa.field("check_id", pa.string(), nullable=False),

    # Raw bank fields
    pa.field("status", pa.string(), nullable=False),          # OPEN / CLEARED
    pa.field("check_type", pa.string()),
    pa.field("source", pa.string()),
    pa.field("post_date", pa.date32(), nullable=False),
    pa.field("amount", pa.float64(), nullable=False),
    pa.field("reference", pa.string()),
    pa.field("bacs_reference", pa.string()),
    pa.field("positive_pay", pa.bool_()),
    pa.field("is_void", pa.bool_()),
    pa.field("is_balanced", pa.bool_()),

    # Clearing info
    pa.field("cleared_flag", pa.bool_()),
    pa.field("cleared_date", pa.date32()),

    # Derived / forecasting fields
    pa.field("days_to_settle", pa.int32()),       # only if CLEARED
    pa.field("current_age_days", pa.int32()),     # only if OPEN
    pa.field("vendor_id", pa.string()),
    pa.field("vendor_name", pa.string()),
    pa.field("cohort", pa.string()),               # STABLE / VOLATILE / LAZY
    pa.field("strategy_used", pa.string()),        # OVERRIDE / SPECIFIC / COHORT
    pa.field("forecast_probability", pa.float64()),
    pa.field("expected_outflow", pa.float64()),

    # Audit
    pa.field("last_updated_run", pa.date32(), nullable=False),
])



# Modeling Constants


COHORT_THRESHOLDS = {
    "SMALL_MAX": 10_000,
    "MED_MAX": 50_000,
}

MIN_HISTORY_FOR_SPECIFIC_MODEL = 5
FORECAST_HORIZON_DAYS = 7
ZOMBIE_CHECK_AGE_DAYS = 45
