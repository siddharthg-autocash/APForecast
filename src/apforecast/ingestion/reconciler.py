# src/apforecast/ingestion/reconciler.py
import pandas as pd
import os
import glob
from src.apforecast.core.constants import *
from src.apforecast.ingestion.loader import load_file_smart

def ingest_and_reconcile(date_str, run_date):
    # 1. Load Brain (Master Ledger)
    if os.path.exists(MASTER_LEDGER_PATH):
        ledger = pd.read_parquet(MASTER_LEDGER_PATH)
    else:
        # --- INITIALIZATION ---
        # Look for ANY file in history folder
        hist_files = glob.glob(f"{DATA_DIR}/raw/history/*")
        if hist_files:
            # Pick the first file found (e.g., your Cleared Checks.csv)
            hist_path = hist_files[0]
            print(f"Initializing Master Ledger from: {hist_path}")
            
            ledger = load_file_smart(hist_path)
            
            # Ensure required columns exist
            required = [COL_CHECK_ID, COL_VENDOR_ID, COL_POST_DATE, COL_CLEAR_DATE]
            if not all(col in ledger.columns for col in required):
                raise ValueError(f"History file missing columns! Found: {ledger.columns}. Check constants.py mapping.")

            ledger[COL_POST_DATE] = pd.to_datetime(ledger[COL_POST_DATE])
            ledger[COL_CLEAR_DATE] = pd.to_datetime(ledger[COL_CLEAR_DATE])
            
            # Calculate history logic
            ledger[COL_DAYS_TO_SETTLE] = (ledger[COL_CLEAR_DATE] - ledger[COL_POST_DATE]).dt.days
            ledger[COL_STATUS] = STATUS_CLEARED
        else:
            raise FileNotFoundError("No history file found in data/raw/history/")

    # 2. Load Daily Inputs
    daily_folder = f"{RAW_DIR}/{date_str}"
    
    # Find the "Outstanding" file (Uncleared)
    input_files = glob.glob(f"{daily_folder}/*")
    
    issued_checks = pd.DataFrame()
    bank_cleared = pd.DataFrame()

    for f in input_files:
        if "Outstanding" in f or "uncleared" in f.lower():
            print(f"Loading Outstanding Checks: {f}")
            issued_checks = load_file_smart(f)
        elif "Cleared" in f or "bank" in f.lower():
            print(f"Loading Cleared Checks: {f}")
            bank_cleared = load_file_smart(f)

    # Standardize IDs
    if COL_CHECK_ID in ledger.columns:
        ledger[COL_CHECK_ID] = ledger[COL_CHECK_ID].astype(str)

    # 3a. Reconciliation (Matches)
    if not bank_cleared.empty and COL_CHECK_ID in bank_cleared.columns:
        bank_cleared[COL_CHECK_ID] = bank_cleared[COL_CHECK_ID].astype(str)
        cleared_ids = bank_cleared[COL_CHECK_ID].tolist()
        
        mask_cleared = ledger[COL_CHECK_ID].isin(cleared_ids)
        ledger.loc[mask_cleared, COL_STATUS] = STATUS_CLEARED
        
        # Update dates
        if COL_CLEAR_DATE in bank_cleared.columns:
            date_map = bank_cleared.set_index(COL_CHECK_ID)[COL_CLEAR_DATE].to_dict()
            ledger.loc[mask_cleared, COL_CLEAR_DATE] = ledger.loc[mask_cleared, COL_CHECK_ID].map(date_map)
            ledger.loc[mask_cleared, COL_CLEAR_DATE] = pd.to_datetime(ledger.loc[mask_cleared, COL_CLEAR_DATE])
        else:
            ledger.loc[mask_cleared, COL_CLEAR_DATE] = run_date
            
        ledger.loc[mask_cleared, COL_DAYS_TO_SETTLE] = (
            ledger.loc[mask_cleared, COL_CLEAR_DATE] - ledger.loc[mask_cleared, COL_POST_DATE]
        ).dt.days

    # 3b. Add New Uncleared Checks
    if not issued_checks.empty and COL_CHECK_ID in issued_checks.columns:
        issued_checks[COL_CHECK_ID] = issued_checks[COL_CHECK_ID].astype(str)
        
        # Filter duplicates
        existing_ids = set(ledger[COL_CHECK_ID])
        new_checks = issued_checks[~issued_checks[COL_CHECK_ID].isin(existing_ids)].copy()
        
        if not new_checks.empty:
            new_checks[COL_STATUS] = STATUS_OPEN
            new_checks[COL_POST_DATE] = pd.to_datetime(new_checks[COL_POST_DATE])
            new_checks[COL_DAYS_TO_SETTLE] = None 
            
            # Align columns before concat
            ledger = pd.concat([ledger, new_checks], ignore_index=True)

    # 4. Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    ledger.to_parquet(MASTER_LEDGER_PATH)
    print("Master Ledger updated.")
    return ledger