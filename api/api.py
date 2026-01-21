# APForecast/api/api.py
# uvicorn api.api:app --reload --port 8010

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
)

app = FastAPI(
    title="APForecast API",
    description="Cash Forecasting API",
    version="1.0.0",
)


def forecast_today(run_date: datetime) -> dict:
    """
    Core forecasting function.
    Returns a JSON-serializable dict.
    """

    # 1. Load & reconcile ledger
    # Note: This loads the file snapshot available for 'run_date'
    ledger = ingest_and_reconcile(run_date.strftime("%Y-%m-%d"), run_date)

    # 2. Train engine (using all available history in that ledger)
    engine = ForecastEngine(ledger)

    # 3. Filter for Open Checks existing ON or BEFORE the run_date
    #    (Enforces strict causality: we can't predict checks not yet posted)
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

    # 4. Predict
    vendor_totals = {}
    total_outstanding = 0.0
    predicted_today = 0.0

    # We use 'run_date - 1 day' as the reference point for probability calculation
    # (i.e., "Given it was open yesterday, what is prob of clearing today?")
    current_date_override = run_date - timedelta(days=1)

    for _, row in open_checks.iterrows():
        amount = float(row[COL_AMOUNT])
        vendor = str(row[COL_VENDOR_ID])

        prob = engine.predict_check(
            row,
            run_date,
            current_date_override=current_date_override,
        )

        expected = amount * prob

        total_outstanding += amount
        predicted_today += expected
        vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + expected

    # ---------------------------------------------------------
    # SORTING & FILTERING
    # ---------------------------------------------------------
    
    # 1. Filter: Keep only vendors with > 0 predicted amount
    # 2. Sort: Descending order by predicted amount
    
    filtered_vendors = {}
    for v, val in vendor_totals.items():
        val_rounded = round(val, 2)
        if val_rounded > 0:
            filtered_vendors[v] = val_rounded
            
    # Sort dictionary by value (descending)
    sorted_vendors = dict(sorted(filtered_vendors.items(), key=lambda item: item[1], reverse=True))

    return {
        "run_date": run_date.date().isoformat(),
        "total_outstanding": round(total_outstanding, 2),
        "predicted_today": round(predicted_today, 2),
        "by_vendor": sorted_vendors,
    }


# -----------------------------
# API ENDPOINTS
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/today")
def forecast_today_endpoint(date: str | None = None):
    """
    GET /forecast/today
    Optional query param: ?date=YYYY-MM-DD
    """

    try:
        run_date = (
            datetime.strptime(date, "%Y-%m-%d")
            if date
            else datetime.today()
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    return forecast_today(run_date)