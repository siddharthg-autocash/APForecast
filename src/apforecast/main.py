# src/apforecast/main.py
"""
Lightweight programmatic interface for APForecast.

Provides:
- forecast_today(run_date_str=None) -> JSON string
  where run_date_str may be e.g. "2026-01-21" or "21-01-2026" (dayfirst accepted).
  If None, uses today's date (system local).
"""

import json
from datetime import datetime, timedelta
import pandas as pd

from src.apforecast.core.constants import *
from src.apforecast.ingestion.reconciler import ingest_and_reconcile
from src.apforecast.modeling.engine import ForecastEngine

def _parse_date(run_date_str):
    """Return a normalized pd.Timestamp or None if parsing fails."""
    if run_date_str is None:
        return pd.Timestamp.today().normalize()
    # Try flexible parsing (dayfirst True to accept DD-MM-YYYY)
    d = pd.to_datetime(run_date_str, dayfirst=True, errors='coerce')
    if pd.isna(d):
        # final fallback: try ISO
        try:
            d = pd.to_datetime(run_date_str)
        except Exception:
            return None
    return d.normalize()

def forecast_today(run_date_str: str = None):
    """
    Build and return a JSON string with:
      - date (YYYY-MM-DD)
      - total_outflow_outstanding (sum of amounts of checks currently OPEN)
      - predicted_today (sum of expected amounts predicted by model for the run_date)
      - by_vendor: { vendor_id: { outstanding: X, predicted: Y } }

    Notes:
    - This function will call ingest_and_reconcile(date_folder_str, run_date_pd).
    """
    run_date_pd = _parse_date(run_date_str)
    if run_date_pd is None:
        return json.dumps({"error": "Invalid run_date_str; could not parse."})

    date_folder_str = run_date_pd.strftime("%Y-%m-%d")

    # Ingest / reconcile - may raise; catch and return structured error
    try:
        ledger = ingest_and_reconcile(date_folder_str, run_date_pd)
    except Exception as e:
        return json.dumps({"error": f"Ingest/Reconcile failed: {str(e)}"})

    # Ensure column names exist
    if COL_STATUS not in ledger.columns or COL_AMOUNT not in ledger.columns:
        return json.dumps({"error": f"Missing required columns in ledger: {list(ledger.columns)}"})

    # Initialize engine (no vendor overrides in this branch)
    engine = ForecastEngine(ledger)

    # ---------------------------------------------------------
    # CAUSALITY FILTER (New)
    # ---------------------------------------------------------
    # 1. Status must be OPEN
    # 2. Post Date must be <= Run Date (Cannot predict checks that don't exist yet)
    mask_open = ledger[COL_STATUS] == STATUS_OPEN
    mask_date = ledger[COL_POST_DATE] <= run_date_pd
    
    open_checks = ledger[mask_open & mask_date].copy()

    # Total outstanding (sum of amounts for open checks)
    total_outstanding = float(open_checks[COL_AMOUNT].sum()) if not open_checks.empty else 0.0

    # Compute predicted expected cash for run_date
    predicted_total = 0.0
    per_vendor = {}

    current_date_override = run_date_pd - pd.Timedelta(days=1)

    for _, row in open_checks.iterrows():
        try:
            prob = engine.predict_check(row, run_date_pd, current_date_override=current_date_override)
        except Exception:
            # If model fails for a row, treat prediction as 0
            prob = 0.0

        expected = float(row[COL_AMOUNT]) * float(prob)
        vendor = str(row.get(COL_VENDOR_ID, "Unknown_Vendor"))

        if vendor not in per_vendor:
            per_vendor[vendor] = {"outstanding": 0.0, "predicted": 0.0}

        per_vendor[vendor]["outstanding"] += float(row[COL_AMOUNT])
        per_vendor[vendor]["predicted"] += expected

        predicted_total += expected

    # ---------------------------------------------------------
    # SORTING & FILTERING LOGIC
    # ---------------------------------------------------------
    
    filtered_vendors_list = []

    for v, data in per_vendor.items():
        p_outstanding = round(data["outstanding"], 2)
        p_predicted = round(data["predicted"], 2)

        # Only include if there is a predicted cash flow > 0
        if p_predicted > 0:
            filtered_vendors_list.append(
                (v, {"outstanding": p_outstanding, "predicted": p_predicted})
            )

    # Sort by 'predicted' in descending order
    filtered_vendors_list.sort(key=lambda item: item[1]["predicted"], reverse=True)

    # Reconstruct dictionary
    by_vendor_sorted = {v: data for v, data in filtered_vendors_list}

    payload = {
        "date": run_date_pd.strftime("%Y-%m-%d"),
        "total_outflow_outstanding": round(total_outstanding, 2),
        "predicted_today": round(predicted_total, 2),
        "by_vendor": by_vendor_sorted
    }

    return json.dumps(payload, indent=2)

# CLI entry
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APForecast - forecast_today JSON output")
    parser.add_argument("--date", type=str, help="Run date (e.g. 2026-01-21). If omitted, uses today.")
    args = parser.parse_args()

    result = forecast_today(args.date)
    print(result)