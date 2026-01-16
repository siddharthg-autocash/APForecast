# src/apforecast/modeling/engine.py

from collections import defaultdict
from datetime import date

import pandas as pd

from apforecast.core.constants import (
    COHORT_THRESHOLDS,
    MIN_HISTORY_FOR_SPECIFIC_MODEL,
    FORECAST_HORIZON_DAYS,
)
from apforecast.modeling.cohorts import assign_cohort
from apforecast.modeling.probability import (
    build_ecdf,
    conditional_clear_probability,
)


# ---------------------------------
# Model Builder
# ---------------------------------

def build_probability_models(ledger_df: pd.DataFrame):
    """
    Build all ECDF models from cleared checks.
    Returns:
        vendor_models: dict[vendor_id -> cdf_fn]
        cohort_models: dict[cohort -> cdf_fn]
    """

    cleared = ledger_df[
        (ledger_df["status"] == "CLEARED")
        & ledger_df["days_to_settle"].notna()
    ]

    # -------------------------
    # Vendor-specific models
    # -------------------------

    vendor_days = defaultdict(list)

    for _, row in cleared.iterrows():
        if row["vendor_id"]:
            vendor_days[row["vendor_id"]].append(row["days_to_settle"])

    vendor_models = {}
    for vendor_id, days in vendor_days.items():
        if len(days) >= MIN_HISTORY_FOR_SPECIFIC_MODEL:
            vendor_models[vendor_id] = build_ecdf(days)

    # -------------------------
    # Cohort models
    # -------------------------

    cohort_days = defaultdict(list)

    for _, row in cleared.iterrows():
        cohort = assign_cohort(row["amount"])
        cohort_days[cohort].append(row["days_to_settle"])

    cohort_models = {
        cohort: build_ecdf(days)
        for cohort, days in cohort_days.items()
        if days
    }

    return vendor_models, cohort_models


# ---------------------------------
# Strategy Selector
# ---------------------------------

def forecast_open_checks(
    ledger_df: pd.DataFrame,
    vendor_models: dict,
    cohort_models: dict,
    override_rules: dict,
    run_date: date,
    window: int = 1,
):
    """
    Apply hybrid strategy to OPEN checks and compute probabilities.
    """

    open_df = ledger_df[ledger_df["status"] == "OPEN"].copy()

    results = []

    for idx, row in open_df.iterrows():
        vendor_id = row["vendor_id"]
        amount = row["amount"]
        age = (run_date - row["post_date"]).days

        # -------------------------
        # Priority 1: Overrides
        # -------------------------

        if vendor_id in override_rules:
            rule = override_rules[vendor_id]
            prob = rule(age, run_date)
            strategy = "OVERRIDE"

        # -------------------------
        # Priority 2: Vendor history
        # -------------------------

        elif vendor_id in vendor_models:
            cdf_fn = vendor_models[vendor_id]
            prob = conditional_clear_probability(
                cdf_fn, age, window
            )
            strategy = "SPECIFIC"

        # -------------------------
        # Priority 3: Cohort model
        # -------------------------

        else:
            cohort = assign_cohort(amount)
            cdf_fn = cohort_models.get(cohort)

            if cdf_fn:
                prob = conditional_clear_probability(
                    cdf_fn, age, window
                )
            else:
                prob = 0.0

            strategy = "COHORT"

        results.append({
            "forecast_probability": prob,
            "expected_outflow": amount * prob,
            "strategy_used": strategy,
        })

    result_df = pd.DataFrame(results, index=open_df.index)

    for col in result_df.columns:
        open_df[col] = result_df[col]

    return open_df

