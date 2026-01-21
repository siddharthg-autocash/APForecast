# src/apforecast/main.py

# python3 -m src.apforecast.main --date 16-01-2026

import argparse
import pandas as pd
import os
from src.apforecast.core.constants import *
from src.apforecast.ingestion.reconciler import ingest_and_reconcile
from src.apforecast.modeling.engine import ForecastEngine
from src.apforecast.reporting.dashboard import generate_report
# --- CHANGE 3: Import Visuals ---
from src.apforecast.reporting.visuals import plot_model_curves

def main():
    parser = argparse.ArgumentParser(description="APForecast System")
    parser.add_argument("--date", type=str, required=True, help="Run date (DD-MM-YYYY)")
    args = parser.parse_args()
    
    run_date_str = args.date
    run_date = pd.to_datetime(run_date_str, format="%d-%m-%Y")
    
    print(f"--- Starting APForecast for {run_date_str} ---")

    # 1. Ingest & Reconcile (Now handles .xlsx)
    ledger = ingest_and_reconcile(run_date_str, run_date)
    
    # 2. Initialize Engine (Training)
    engine = ForecastEngine(ledger)
    
    # --- CHANGE 4: Generate Reference Graphs ---
    # This will create the PNGs for every vendor/cohort
    plot_model_curves(engine.models, run_date_str)
    
    # 4. Forecast Loop
    open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()
    print(f"Forecasting for {len(open_checks)} open checks...")
    
    results = []
    for _, row in open_checks.iterrows():
        prob = engine.predict_check(row, run_date)
        expected_cash = row[COL_AMOUNT] * prob
        
        results.append({
            COL_CHECK_ID: row[COL_CHECK_ID],
            COL_VENDOR_ID: row[COL_VENDOR_ID],
            COL_AMOUNT: row[COL_AMOUNT],
            COL_POST_DATE: row[COL_POST_DATE],
            'Probability': round(prob, 4),
            'Expected_Cash': round(expected_cash, 2)
        })
        
    forecast_df = pd.DataFrame(results)
    
    # 5. Report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    generate_report(forecast_df, run_date_str)
    
    print("--- Process Complete ---")

if __name__ == "__main__":
    main()
