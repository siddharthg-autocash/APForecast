# experiments/backtesting/core.py

import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.apforecast.core.constants import *
from src.apforecast.modeling.engine import ForecastEngine

def run_walk_forward_backtest(full_ledger, start_date, end_date, vendor_filter=None):
    """
    Rolling Backtest Logic (final):

    - Trains & predicts on business days (US Federal calendar).
    - Aggregates clears that occurred on non-business days to the *next* business day for actuals.
    - If the model predicts a check will clear with ~100% probability on a day, that check is
      removed from all subsequent prediction days (so it doesn't reappear).
    - Returns a full calendar series (daily rows). Non-business days show Predicted=0, Actual=0.
    """
    full_ledger = full_ledger.copy()

    # Normalize dates
    full_ledger[COL_CLEAR_DATE] = pd.to_datetime(full_ledger[COL_CLEAR_DATE], errors='coerce')
    full_ledger[COL_POST_DATE] = pd.to_datetime(full_ledger[COL_POST_DATE], errors='coerce')

    # Mark cleared status
    valid_dates = full_ledger[COL_CLEAR_DATE].notna()
    full_ledger.loc[valid_dates, COL_STATUS] = STATUS_CLEARED

    # Vendor filter
    if vendor_filter:
        print(f"🎯 Filtered Backtest for Vendor: {vendor_filter}")
        full_ledger = full_ledger[full_ledger[COL_VENDOR_ID] == vendor_filter].copy()
        if full_ledger.empty:
            return pd.DataFrame()

    # PREPROCESS: map non-business Clear_Date -> next business day for backtest actual aggregation
    full_ledger['Clear_Date_for_backtest'] = full_ledger[COL_CLEAR_DATE].apply(
        lambda d: pd.NaT if pd.isna(d) else (d if is_business_day(d) else next_business_day(d))
    )

    # Business-day index used for training & prediction calculation
    business_days = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq=USB)
    print(f"🔄 Starting Walk-Forward Backtest ({start_date} to {end_date}) on {len(business_days)} business days...")

    # Results for business days
    business_results = []
    cumulative_error = 0.0

    # Track IDs that model has effectively 'deterministically predicted' to clear so they are
    # excluded from subsequent predictions.
    predicted_cleared_ids = set()
    PRED_DET_EPS = 1.0 - 1e-9  # threshold to treat as effectively 100%

    for current_date in business_days:
        # --- TRAIN ---
        history_mask = (full_ledger[COL_STATUS] == STATUS_CLEARED) & (full_ledger[COL_CLEAR_DATE] < current_date)
        training_data = full_ledger[history_mask].copy()

        min_history = 5 if vendor_filter else 50
        if len(training_data) < min_history:
            # Not enough history — produce a zero row for this business day (still include in business_results)
            business_results.append({
                'Date': current_date,
                'Predicted': 0.0,
                'Actual': full_ledger[pd.to_datetime(full_ledger['Clear_Date_for_backtest']) == pd.to_datetime(current_date)][COL_AMOUNT].sum(),
                'Residual ($)': 0.0,
                'Error %': 0.0,
                'Cum. Error': cumulative_error,
                'Flagged (>45d)': 0.0
            })
            continue

        engine = ForecastEngine(training_data)

        # Identify open checks as of this business day (only consider those not already actually cleared BEFORE current_date)
        open_mask = (full_ledger[COL_POST_DATE] <= current_date) & (
            (full_ledger[COL_CLEAR_DATE].isna()) | (full_ledger[COL_CLEAR_DATE] >= current_date)
        )
        daily_open_checks = full_ledger[open_mask].copy()

        predicted_cash = 0.0
        flagged_volume = 0.0

        # Iterate checks and predict; skip checks already predicted deterministically earlier
        for _, row in daily_open_checks.iterrows():
            cid = str(row[COL_CHECK_ID])
            if cid in predicted_cleared_ids:
                # previously predicted to be 100% -> remove from future predictions
                continue

            age = business_days_between(row[COL_POST_DATE], current_date)
            if age > 45:
                flagged_volume += row[COL_AMOUNT]
                continue

            prob = engine.predict_check(
                row,
                current_date,
                current_date_override=prev_business_day(current_date),
            )

            # If model thinks this clears with (virtually) certainty today, remove it for future days
            if prob >= PRED_DET_EPS:
                predicted_cleared_ids.add(cid)

            predicted_cash += (row[COL_AMOUNT] * prob)

        # ACTUALS: aggregated to Clear_Date_for_backtest
        actual_mask = pd.to_datetime(full_ledger['Clear_Date_for_backtest']) == pd.to_datetime(current_date)
        actual_cash = full_ledger[actual_mask][COL_AMOUNT].sum()

        # METRICS
        residual = actual_cash - predicted_cash
        cumulative_error += residual
        error_pct = (residual / actual_cash) if actual_cash > 10 else 0.0

        business_results.append({
            'Date': current_date,
            'Predicted': round(predicted_cash, 2),
            'Actual': round(actual_cash, 2),
            'Residual ($)': round(residual, 2),
            'Error %': round(error_pct * 100, 1),
            'Cum. Error': round(cumulative_error, 2),
            'Flagged (>45d)': round(flagged_volume, 2)
        })

    # Build a DataFrame for business days
    df_business = pd.DataFrame(business_results)
    if df_business.empty:
        # Return an all-zero calendar frame if nothing computed
        calendar = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='D')
        rows = []
        cum_err = 0.0
        for d in calendar:
            rows.append({'Date': d, 'Predicted': 0.0, 'Actual': 0.0, 'Residual ($)': 0.0, 'Error %': 0.0, 'Cum. Error': cum_err, 'Flagged (>45d)': 0.0})
        return pd.DataFrame(rows)

    # Create full calendar series (daily), inserting business-day rows and zero rows for non-business days
    calendar = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='D')
    business_dates_map = {pd.to_datetime(r['Date']).normalize(): r for _, r in df_business.iterrows()}

    rows = []
    last_cum_error = 0.0
    for d in calendar:
        dn = pd.to_datetime(d).normalize()
        if dn in business_dates_map:
            r = business_dates_map[dn]
            last_cum_error = r['Cum. Error']
            rows.append({
                'Date': dn,
                'Predicted': r['Predicted'],
                'Actual': r['Actual'],
                'Residual ($)': r['Residual ($)'],
                'Error %': r['Error %'],
                'Cum. Error': r['Cum. Error'],
                'Flagged (>45d)': r.get('Flagged (>45d)', 0.0)
            })
        else:
            # Non-business day: keep zeros for both Predicted & Actual (so RMSE includes zero-zero days)
            rows.append({
                'Date': dn,
                'Predicted': 0.0,
                'Actual': 0.0,
                'Residual ($)': 0.0,
                'Error %': 0.0,
                'Cum. Error': last_cum_error,
                'Flagged (>45d)': 0.0
            })

    full_df = pd.DataFrame(rows)
    return full_df

def plot_backtest_results(df, title_prefix="Global"):
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f"{title_prefix} Cash Flow: Predicted vs Actual", "Daily Deviation (Residual Error)"),
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual'], name='Actual Cash',
        line=dict(color='#27AE60', width=2), fill='tozeroy', opacity=0.2
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Predicted'], name='Model Prediction',
        mode='lines+markers', line=dict(color='#154360', width=3), marker=dict(size=4)
    ), row=1, col=1)

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
