# experiments/backtesting/core.py
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.apforecast.core.constants import *
from src.apforecast.modeling.engine import ForecastEngine

def run_walk_forward_backtest(full_ledger, start_date, end_date, overrides, vendor_filter=None):
    """
    Rolling Backtest Logic.
    - Added 'vendor_filter': If set, runs backtest ONLY for that vendor.
    """
    
    # 1. FORCE STATUS Logic (Fix missing 'CLEARED' labels)
    full_ledger = full_ledger.copy()
    valid_dates = pd.to_datetime(full_ledger['Clear_Date'], errors='coerce').notna()
    full_ledger.loc[valid_dates, 'Status'] = 'CLEARED'
    
    # 2. FILTER BY VENDOR (If requested)
    if vendor_filter:
        print(f"🎯 Filtered Backtest for Vendor: {vendor_filter}")
        full_ledger = full_ledger[full_ledger[COL_VENDOR_ID] == vendor_filter].copy()
        if full_ledger.empty:
            return pd.DataFrame()

    results = []
    current_date = pd.to_datetime(start_date)
    end_date_pd = pd.to_datetime(end_date)
    
    print(f"🔄 Starting Walk-Forward Backtest ({start_date} to {end_date})...")
    
    cumulative_error = 0.0

    while current_date <= end_date_pd:
        # --- TRAIN ---
        history_mask = (full_ledger['Status'] == 'CLEARED') & (full_ledger['Clear_Date'] < current_date)
        training_data = full_ledger[history_mask].copy()
        
        # Lower threshold for specific vendor backtests
        min_history = 5 if vendor_filter else 50
        
        if len(training_data) < min_history:
            # print(f"⚠️ Skipping {current_date.date()}: Not enough history.")
            current_date += timedelta(days=1)
            continue

        engine = ForecastEngine(training_data, overrides)
        
        # --- IDENTIFY OPEN CHECKS ---
        open_mask = (full_ledger[COL_POST_DATE] <= current_date) & (full_ledger['Clear_Date'] >= current_date)
        daily_open_checks = full_ledger[open_mask].copy()
        
        # --- PREDICT ---
        predicted_cash = 0.0
        flagged_volume = 0.0 
        
        for _, row in daily_open_checks.iterrows():
            age = (current_date - row[COL_POST_DATE]).days
            if age > 45:
                flagged_volume += row[COL_AMOUNT]
                continue
            
            prob = engine.predict_check(
                row, 
                current_date, 
                current_date_override=current_date - timedelta(days=1)
            )
            predicted_cash += (row[COL_AMOUNT] * prob)
            
        # --- ACTUALS ---
        actual_mask = (full_ledger['Clear_Date'] == current_date)
        actual_cash = full_ledger[actual_mask][COL_AMOUNT].sum()
        
        # --- METRICS ---
        residual = actual_cash - predicted_cash
        cumulative_error += residual
        error_pct = (residual / actual_cash) if actual_cash > 10 else 0.0
        
        results.append({
            'Date': current_date,
            'Predicted': round(predicted_cash, 2),
            'Actual': round(actual_cash, 2),
            'Residual ($)': round(residual, 2),
            'Error %': round(error_pct * 100, 1),
            'Cum. Error': round(cumulative_error, 2),
            'Flagged (>45d)': round(flagged_volume, 2)
        })
        
        current_date += timedelta(days=1)
        
    return pd.DataFrame(results)

def plot_backtest_results(df, title_prefix="Global"):
    """
    Generates the Perfect 2-Panel Dashboard.
    """
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f"{title_prefix} Cash Flow: Predicted vs Actual", "Daily Deviation (Residual Error)"),
        row_heights=[0.7, 0.3]
    )
    
    # Panel 1: Main
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual'], name='Actual Cash',
        line=dict(color='#27AE60', width=2), fill='tozeroy', opacity=0.2
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Predicted'], name='Model Prediction',
        mode='lines+markers', line=dict(color='#154360', width=3), marker=dict(size=4)
    ), row=1, col=1)
    
    # Panel 2: Errors
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Residual ($)'], name='Error ($)',
        marker_color='crimson', opacity=0.6,
        hovertemplate='%{y:$.2f}<br>Error: %{customdata}%',
        customdata=df['Error %']
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"<b>{title_prefix} Backtest Report</b>",
        template="plotly_white", height=700, hovermode="x unified"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    return fig