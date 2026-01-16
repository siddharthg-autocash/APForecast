# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

# Import your existing "Brain" logic
from src.apforecast.core.constants import *
from src.apforecast.ingestion.reconciler import ingest_and_reconcile
from src.apforecast.modeling.engine import ForecastEngine
from src.apforecast.core.config_loader import load_vendor_overrides
from src.apforecast.reporting.dashboard import generate_report

# --- HELPER: PLOTTING ---
def plot_vendor_behavior(model, vendor_name):
    """Generates the Reference Graph (Orange/Blue) on the fly for Streamlit."""
    if model.n < 2:
        st.warning(f"Not enough data to plot behavior for {vendor_name}")
        return
    
    data = model.sorted_data
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Colors
    BAR_COLOR = '#E67E22'  # Burnt Orange
    LINE_COLOR = '#154360' # Dark Slate Blue
    
    # Histogram
    sns.histplot(data, bins=range(0, int(max(data))+5), color=BAR_COLOR, alpha=0.7, ax=ax1, stat='count', edgecolor=None)
    ax1.set_ylabel('Frequency', color=BAR_COLOR, fontweight='bold')
    ax1.set_xlabel('Days to Settle', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=BAR_COLOR)
    
    # CDF
    ax2 = ax1.twinx()
    x_vals = np.linspace(0, max(data)+5, 100)
    y_vals = [model.cdf(x) for x in x_vals]
    ax2.plot(x_vals, y_vals, color=LINE_COLOR, linewidth=3)
    ax2.set_ylabel('Probability', color=LINE_COLOR, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor=LINE_COLOR)
    
    plt.title(f"Behavior Pattern: {vendor_name}", fontsize=12)
    plt.grid(True, alpha=0.2)
    st.pyplot(fig)

# --- APP CONFIG ---
st.set_page_config(page_title="APForecast Commander", layout="wide", page_icon="💸")
st.title("💸 APForecast Commander")

# --- SESSION STATE INITIALIZATION ---
if 'forecast_data' not in st.session_state:
    st.session_state['forecast_data'] = None
if 'ledger' not in st.session_state:
    st.session_state['ledger'] = None

# --- SIDEBAR ---
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to:", [
    "🚀 Daily Forecast", 
    "🕵️ Vendor Intelligence", 
    "📚 History Setup"
])

# ==========================================
# MODE 1: DAILY FORECAST
# ==========================================
if mode == "🚀 Daily Forecast":
    st.header("🚀 Daily Cash Forecast")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        run_date = st.date_input("Today's Date", value="today")
        run_date_str = run_date.strftime("%d-%m-%Y")
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 14, 7)

    st.info("Drop your files here to update the brain and generate numbers.")
    c1, c2 = st.columns(2)
    with c1:
        cleared_file = st.file_uploader("Checks Cleared Yesterday (Optional)", type=['xlsx', 'csv'])
    with c2:
        uncleared_file = st.file_uploader("Outstanding Checks (Required)", type=['xlsx', 'csv'])

    if st.button("RUN FORECAST", type="primary"):
        if not uncleared_file:
            st.error("❌ You must upload 'Outstanding Checks'!")
        else:
            with st.spinner("🧠 Converting Excel to Cash Logic..."):
                # 1. Save Files
                daily_dir = f"{RAW_DIR}/{run_date_str}"
                os.makedirs(daily_dir, exist_ok=True)
                
                if cleared_file:
                    with open(f"{daily_dir}/bank_cleared.xlsx", "wb") as f:
                        f.write(cleared_file.getbuffer())
                
                with open(f"{daily_dir}/uncleared_checks.xlsx", "wb") as f:
                    f.write(uncleared_file.getbuffer())
                
                # 2. Run Engine
                try:
                    run_date_pd = pd.to_datetime(run_date)
                    ledger = ingest_and_reconcile(run_date_str, run_date_pd)
                    st.session_state['ledger'] = ledger 
                    
                    overrides = load_vendor_overrides()
                    engine = ForecastEngine(ledger, overrides)
                    
                    # 3. Predict (Multi-Day Logic for Screen)
                    open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()
                    detailed_forecast = []
                    daily_totals = { (run_date_pd + timedelta(days=i)).strftime("%Y-%m-%d"): 0 for i in range(forecast_horizon) }

                    # --- NEW STEP 4: GENERATE EXCEL REPORT (Physical File) ---
                    # We create the standard "Snapshot" DataFrame for the Excel report
                    snapshot_data = []
                    for _, row in open_checks.iterrows():
                        # Calculate probability for TODAY (Day 0) for the report
                        prob_today = engine.predict_check(row, run_date_pd)
                        snapshot_data.append({
                            COL_CHECK_ID: row[COL_CHECK_ID],
                            COL_VENDOR_ID: row[COL_VENDOR_ID],
                            COL_AMOUNT: row[COL_AMOUNT],
                            COL_POST_DATE: row[COL_POST_DATE],
                            'Probability': round(prob_today, 4),
                            'Expected_Cash': round(row[COL_AMOUNT] * prob_today, 2)
                        })
                    
                    # Save the Excel File
                    snapshot_df = pd.DataFrame(snapshot_data)
                    os.makedirs(REPORTS_DIR, exist_ok=True)
                    generate_report(snapshot_df, run_date_str)
                    st.success(f"📄 Excel Report saved to: reports/forecast_{run_date_str}.xlsx")
                    # -------------------------------------------------------

                    # Continue with Screen Logic (Multi-Day)
                    for _, row in open_checks.iterrows():
                        for i in range(forecast_horizon):
                            target_date = run_date_pd + timedelta(days=i)
                            target_str = target_date.strftime("%Y-%m-%d")
                            
                            prob_cum_today = engine.predict_check(row, target_date)
                            prob_cum_yesterday = engine.predict_check(row, target_date - timedelta(days=1))
                            prob_marginal = max(0, prob_cum_today - prob_cum_yesterday)
                            
                            exp_cash = row[COL_AMOUNT] * prob_marginal
                            daily_totals[target_str] += exp_cash
                            
                            if exp_cash > 0.01:
                                detailed_forecast.append({
                                    "Date": target_str,
                                    "Amount": exp_cash,
                                    "Vendor": row[COL_VENDOR_ID],
                                    "Check": row[COL_CHECK_ID]
                                })
                    
                    st.session_state['forecast_data'] = pd.DataFrame(detailed_forecast)
                    
                    # Dashboard
                    st.markdown("---")
                    today_str = run_date_pd.strftime("%Y-%m-%d")
                    today_cash = daily_totals.get(today_str, 0)
                    st.metric("CASH REQUIRED TODAY", f"${today_cash:,.2f}")
                    
                    df_chart = pd.DataFrame(list(daily_totals.items()), columns=['Date', 'Expected Outflow'])
                    st.bar_chart(df_chart.set_index('Date'))
                    
                    # Download Button
                    with open(f"reports/forecast_{run_date_str}.xlsx", "rb") as file:
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=file,
                            file_name=f"forecast_{run_date_str}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# MODE 2: VENDOR INTELLIGENCE
# ==========================================
elif mode == "🕵️ Vendor Intelligence":
    st.header("🕵️ Vendor Intelligence")
    
    # Load Ledger if not in session (from disk)
    if st.session_state['ledger'] is None:
        if os.path.exists(MASTER_LEDGER_PATH):
            st.session_state['ledger'] = pd.read_parquet(MASTER_LEDGER_PATH)
        else:
            st.warning("⚠️ No Master Ledger found. Please go to 'History Setup' or Run a Forecast first.")
            st.stop()
            
    ledger = st.session_state['ledger']
    
    # Initialize Engine to get Models
    overrides = load_vendor_overrides()
    engine = ForecastEngine(ledger, overrides)
    
    # Vendor Selector
    all_vendors = sorted(ledger[COL_VENDOR_ID].astype(str).unique())
    selected_vendor = st.selectbox("Select a Vendor to Analyze:", all_vendors)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Reference Behavior (The Brain)")
        st.caption("This is how this vendor has behaved in the past.")
        
        # Check if we have a specific model or using global
        if selected_vendor in engine.models['SPECIFIC']:
            model = engine.models['SPECIFIC'][selected_vendor]
            st.success("✅ Using Specific Vendor Model")
            plot_vendor_behavior(model, selected_vendor)
        else:
            st.warning("⚠️ Using Global Cohort Model (Not enough specific history)")
            # You could plot the cohort model here if you want
            
    with col2:
        st.subheader("2. Current Forecast (The Future)")
        st.caption("Projected cash outflow for this vendor in the current batch.")
        
        if st.session_state['forecast_data'] is not None:
            df_forecast = st.session_state['forecast_data']
            # Filter for this vendor
            vendor_future = df_forecast[df_forecast['Vendor'] == selected_vendor]
            
            if not vendor_future.empty:
                # Group by Date
                daily_sums = vendor_future.groupby('Date')['Amount'].sum().reset_index()
                st.bar_chart(daily_sums.set_index('Date'))
                st.dataframe(daily_sums.style.format({"Amount": "${:,.2f}"}))
            else:
                st.info("No outstanding checks forecasted for this vendor in the current batch.")
        else:
            st.info("Run a 'Daily Forecast' first to see future predictions.")

# ==========================================
# MODE 3: HISTORY SETUP
# ==========================================
elif mode == "📚 History Setup":
    st.header("📚 History Setup")
    st.file_uploader("Upload Historical Cleared Checks", type=['xlsx', 'csv'])
    st.button("Ingest History (Dummy Button for now)")