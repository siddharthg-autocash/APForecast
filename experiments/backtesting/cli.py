# experiments/backtesting/cli.py
import sys
import os
import argparse
import pandas as pd
from datetime import datetime

# --- PATH HACK (To import 'src') ---
current_file = os.path.abspath(__file__)
backtesting_dir = os.path.dirname(current_file) 
experiments_dir = os.path.dirname(backtesting_dir) 
project_root = os.path.dirname(experiments_dir) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------

# Split imports correctly
from src.apforecast.core.constants import * from src.apforecast.core.config_loader import load_vendor_overrides
from experiments.backtesting.core import run_walk_forward_backtest, plot_backtest_results

def normalize_columns(df):
    """
    Maps various Excel header styles to the internal standard names.
    """
    column_map = {
        # Clear Date Variations
        "Clear Date": "Clear_Date",
        "ClearDate": "Clear_Date",
        "Cleared Date": "Clear_Date",
        
        # Post Date Variations
        "Post Date": COL_POST_DATE,
        "PostDate": COL_POST_DATE,
        "Date": COL_POST_DATE,
        
        # Vendor Variations
        "Vendor ID": COL_VENDOR_ID,
        "Vendor Name": COL_VENDOR_ID,
        "Vendor": COL_VENDOR_ID,
        "Payee": COL_VENDOR_ID,
        
        # Amount Variations
        "Amount": COL_AMOUNT,
        "Check Amount": COL_AMOUNT,
        "Transaction Amount": COL_AMOUNT,
        
        # Check ID Variations
        "Check Number": COL_CHECK_ID,
        "Check #": COL_CHECK_ID,
        "Reference": COL_CHECK_ID
    }
    
    df = df.rename(columns=column_map)
    
    # --- CRITICAL FIX: Assign Dummy Vendor BEFORE Validation ---
    if COL_VENDOR_ID not in df.columns:
        print("⚠️ WARNING: No Vendor ID found. Assigning 'Unknown_Vendor' to all rows.")
        df[COL_VENDOR_ID] = "Unknown_Vendor"
    # -----------------------------------------------------------
    
    required = ["Clear_Date", COL_POST_DATE, COL_AMOUNT, COL_VENDOR_ID]
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        raise ValueError(f"❌ Missing critical columns: {missing}\n   Found: {list(df.columns)}")
        
    return df

def main():
    parser = argparse.ArgumentParser(description="🧪 Rolling Backtest Verification")
    parser.add_argument("--start", type=str, required=True, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End Date (YYYY-MM-DD)")
    parser.add_argument("--source", type=str, required=True, help="Path to historical file")
    
    args = parser.parse_args()
    
    print("\n🧪 INITIALIZING ROLLING BACKTEST LAB")
    print("=" * 60)
    
    try:
        print(f"📂 Loading History from: {args.source}")
        df = pd.read_excel(args.source)
        df = normalize_columns(df)
        
        # Convert Dates
        df[COL_POST_DATE] = pd.to_datetime(df[COL_POST_DATE])
        df["Clear_Date"] = pd.to_datetime(df["Clear_Date"])
        
        # Ensure Status exists
        if COL_STATUS not in df.columns:
            df[COL_STATUS] = 'CLEARED'

        # === NEW: DATA DIAGNOSTICS ===
        min_date = df['Clear_Date'].min()
        max_date = df['Clear_Date'].max()
        count = len(df)
        
        print(f"✅ Loaded {count} records.")
        print("-" * 30)
        print(f"📅 EARLIEST Clear Date: {min_date.date()}")
        print(f"📅 LATEST   Clear Date: {max_date.date()}")
        print("-" * 30)
        
        # Validation Logic
        start_dt = pd.to_datetime(args.start)
        
        # Count how many records exist BEFORE the requested start date
        prior_history = df[df['Clear_Date'] < start_dt]
        print(f"🔍 History available before {args.start}: {len(prior_history)} records")
        
        if len(prior_history) < 50:
            print("\n❌ CRITICAL ISSUE: Not enough history to train the model.")
            print(f"   You requested start date: {args.start}")
            print(f"   But your data only starts at: {min_date.date()}")
            print("   👉 ACTION: Please choose a '--start' date at least 3 months AFTER your earliest data.")
            return
        # ==============================

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Run Backtest
    try:
        overrides = load_vendor_overrides()
        metrics_df = run_walk_forward_backtest(df, args.start, args.end, overrides)
        
        if metrics_df.empty:
            print("❌ No results generated.")
            return

        total_pred = metrics_df['Predicted'].sum()
        total_act = metrics_df['Actual'].sum()
        total_err = total_pred - total_act
        
        print("\n📊 SUMMARY RESULTS")
        print("-" * 30)
        print(f"📅 Period: {args.start} to {args.end}")
        print(f"💰 Total Predicted: ${total_pred:,.2f}")
        print(f"💰 Total Actual:    ${total_act:,.2f}")
        print(f"📉 Net Variance:    ${total_err:,.2f} ({total_err/total_act:.1%})")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save Outputs
    output_dir = os.path.join(project_root, "experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    excel_path = os.path.join(output_dir, f"backtest_metrics_{timestamp}.xlsx")
    metrics_df.to_excel(excel_path, index=False)
    
    fig = plot_backtest_results(metrics_df)
    html_path = os.path.join(output_dir, f"backtest_plot_{timestamp}.html")
    fig.write_html(html_path)
    
    print(f"\n✅ Verification Complete.")
    print(f"📝 Excel: {excel_path}")
    print(f"📈 Graph: {html_path}\n")

if __name__ == "__main__":
    main()