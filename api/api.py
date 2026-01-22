# api/api.py
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import pandas as pd

from src.apforecast.ingestion.reconciler import ingest_and_reconcile
from src.apforecast.modeling.engine import ForecastEngine
from src.apforecast.core.constants import (
    COL_VENDOR_ID,
    COL_AMOUNT,
    COL_POST_DATE,
    COL_STATUS,
    STATUS_OPEN,
    is_business_day,
    next_business_day,
    prev_business_day,
)

app = FastAPI(
    title="APForecast API",
    description="Cash Forecasting API",
    version="1.0.0",
)

def forecast_today(run_date: datetime) -> dict:
    ledger = ingest_and_reconcile(run_date.strftime("%Y-%m-%d"), run_date)
    engine = ForecastEngine(ledger)

    mask_open = ledger[COL_STATUS] == STATUS_OPEN
    mask_date = ledger[COL_POST_DATE] <= run_date
    open_checks = ledger[mask_open & mask_date].copy()

    if open_checks.empty:
        return {
            "run_date": run_date.date().isoformat(),
            "total_outstanding": 0.0,
            "predicted_today": 0.0,
            "by_vendor": {},
        }

    vendor_totals = {}
    total_outstanding = 0.0
    predicted_today = 0.0
    predicted_next_bd = 0.0

    if is_business_day(run_date):
        effective_today = run_date
        next_bd = None
    else:
        effective_today = None
        next_bd = next_business_day(run_date)

    current_date_override_for_next = prev_business_day(next_bd) if next_bd is not None else None
    current_date_override_for_today = prev_business_day(effective_today) if effective_today is not None else None

    for _, row in open_checks.iterrows():
        amount = float(row[COL_AMOUNT])
        vendor = str(row[COL_VENDOR_ID])

        if next_bd is not None:
            prob_next = engine.predict_check(row, next_bd, current_date_override=current_date_override_for_next)
            expected_next = amount * prob_next
            predicted_next_bd += expected_next
            vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + expected_next

        if effective_today is not None:
            prob_today = engine.predict_check(row, effective_today, current_date_override=current_date_override_for_today)
            expected_today = amount * prob_today
            predicted_today += expected_today
            vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + expected_today

        total_outstanding += amount

    filtered_vendors = {}
    for v, val in vendor_totals.items():
        val_rounded = round(val, 2)
        if val_rounded > 0:
            filtered_vendors[v] = val_rounded

    sorted_vendors = dict(sorted(filtered_vendors.items(), key=lambda item: item[1], reverse=True))

    if is_business_day(run_date):
        return {
            "run_date": run_date.date().isoformat(),
            "total_outstanding": round(total_outstanding, 2),
            "predicted_today": round(predicted_today, 2),
            "by_vendor": sorted_vendors,
        }
    else:
        return {
            "run_date": run_date.date().isoformat(),
            "total_outstanding": round(total_outstanding, 2),
            "predicted_today": 0.0,
            "next_business_day": next_bd.date().isoformat(),
            "predicted_next_business_day": round(predicted_next_bd, 2),
            "by_vendor": sorted_vendors,
        }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast/today")
def forecast_today_endpoint(date: str | None = None):
    try:
        run_date = (
            datetime.strptime(date, "%Y-%m-%d")
            if date
            else datetime.today()
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    return forecast_today(run_date)
