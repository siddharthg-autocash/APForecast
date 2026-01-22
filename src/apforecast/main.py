# src/apforecast/main.py
import json
from datetime import datetime, timedelta
import pandas as pd

from src.apforecast.core.constants import *
from src.apforecast.ingestion.reconciler import ingest_and_reconcile
from src.apforecast.modeling.engine import ForecastEngine

def _parse_date(run_date_str):
    if run_date_str is None:
        return pd.Timestamp.today().normalize()
    d = pd.to_datetime(run_date_str, dayfirst=True, errors='coerce')
    if pd.isna(d):
        try:
            d = pd.to_datetime(run_date_str)
        except Exception:
            return None
    return d.normalize()

def forecast_today(run_date_str: str = None):
    run_date_pd = _parse_date(run_date_str)
    if run_date_pd is None:
        return json.dumps({"error": "Invalid run_date_str; could not parse."})

    date_folder_str = run_date_pd.strftime("%Y-%m-%d")

    try:
        ledger = ingest_and_reconcile(date_folder_str, run_date_pd)
    except Exception as e:
        return json.dumps({"error": f"Ingest/Reconcile failed: {str(e)}"})

    if COL_STATUS not in ledger.columns or COL_AMOUNT not in ledger.columns:
        return json.dumps({"error": f"Missing required columns in ledger: {list(ledger.columns)}"})

    engine = ForecastEngine(ledger)

    mask_open = ledger[COL_STATUS] == STATUS_OPEN
    mask_date = ledger[COL_POST_DATE] <= run_date_pd
    open_checks = ledger[mask_open & mask_date].copy()

    total_outstanding = float(open_checks[COL_AMOUNT].sum()) if not open_checks.empty else 0.0

    predicted_total = 0.0
    per_vendor = {}

    if is_business_day(run_date_pd):
        effective_today = run_date_pd
        next_bd = None
    else:
        effective_today = None
        next_bd = next_business_day(run_date_pd)

    predicted_today = 0.0
    predicted_next_bd = 0.0

    current_date_override_for_next = prev_business_day(next_bd) if next_bd is not None else None
    current_date_override_for_today = prev_business_day(effective_today) if effective_today is not None else None

    for _, row in open_checks.iterrows():
        try:
            vendor = str(row.get(COL_VENDOR_ID, "Unknown_Vendor"))
            amount = float(row[COL_AMOUNT])
        except Exception:
            continue

        if next_bd is not None:
            prob_next = engine.predict_check(row, next_bd, current_date_override=current_date_override_for_next)
            expected_next = amount * prob_next
            predicted_next_bd += expected_next
            if vendor not in per_vendor:
                per_vendor[vendor] = {"outstanding": 0.0, "predicted": 0.0}
            per_vendor[vendor]["outstanding"] += amount
            per_vendor[vendor]["predicted"] += expected_next

        if effective_today is not None:
            prob_today = engine.predict_check(row, effective_today, current_date_override=current_date_override_for_today)
            expected = amount * prob_today
            predicted_today += expected

    payload = {
        "date": run_date_pd.strftime("%Y-%m-%d"),
        "total_outflow_outstanding": round(total_outstanding, 2),
        "predicted_today": round(predicted_today, 2) if is_business_day(run_date_pd) else 0.0,
        "by_vendor": {}
    }

    filtered_vendors_list = []
    for v, data in per_vendor.items():
        p_outstanding = round(data["outstanding"], 2)
        p_predicted = round(data["predicted"], 2)
        if p_predicted > 0:
            filtered_vendors_list.append((v, {"outstanding": p_outstanding, "predicted": p_predicted}))
    filtered_vendors_list.sort(key=lambda item: item[1]["predicted"], reverse=True)

    payload["by_vendor"] = {v: data for v, data in filtered_vendors_list}

    if not is_business_day(run_date_pd):
        payload["next_business_day"] = next_bd.strftime("%Y-%m-%d")
        payload["predicted_next_business_day"] = round(predicted_next_bd, 2)

    return json.dumps(payload, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APForecast - forecast_today JSON output")
    parser.add_argument("--date", type=str, help="Run date (e.g. 2026-01-21). If omitted, uses today.")
    args = parser.parse_args()

    result = forecast_today(args.date)
    print(result)
