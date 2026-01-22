# ====================
# FILE: ./app.py
# ====================

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import sys
import plotly.graph_objects as go
from datetime import datetime, timedelta

from scipy.interpolate import make_interp_spline

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from src.apforecast.core.constants import *
    from src.apforecast.ingestion.reconciler import ingest_and_reconcile
    from src.apforecast.modeling.engine import ForecastEngine
    from experiments.backtesting.core import run_walk_forward_backtest, plot_backtest_results
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="APForecast Commander", layout="wide", page_icon="💸")
st.title("💸 APForecast Commander")

# --- SESSION STATE ---
if 'ledger' not in st.session_state: st.session_state['ledger'] = None
if 'forecast_df' not in st.session_state: st.session_state['forecast_df'] = None
if 'bt_clean' not in st.session_state: st.session_state['bt_clean'] = None
if 'bt_dates' not in st.session_state: st.session_state['bt_dates'] = None
if 'bt_res' not in st.session_state: st.session_state['bt_res'] = None
if 'predicted_total' not in st.session_state: st.session_state['predicted_total'] = None
if 'carried_to_next_bd' not in st.session_state: st.session_state['carried_to_next_bd'] = None

# ==========================================
# 0. DATA UTILITIES
# ==========================================
def smart_normalize_columns(df):
    """
    Robustly finds columns.
    PRIORITY: Forces 'Reference' (Column G) to be Vendor if present.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    found_vendor = False

    for col in df.columns:
        c_lower = col.lower()
        if 'reference' in c_lower and 'bacs' not in c_lower:
            rename_map[col] = COL_VENDOR_ID
            found_vendor = True
            break

    for col in df.columns:
        if col in rename_map: continue
        c_lower = col.lower()

        if not found_vendor:
            if any(x in c_lower for x in ['vendor', 'payee', 'beneficiary', 'name', 'description']):
                rename_map[col] = COL_VENDOR_ID
                found_vendor = True
                continue

        if any(x in c_lower for x in ['amount', 'debit', 'payment']):
            if COL_AMOUNT not in rename_map.values():
                rename_map[col] = COL_AMOUNT
                continue
        if any(x in c_lower for x in ['check', 'num', 'doc']):
            if 'amount' not in c_lower and COL_CHECK_ID not in rename_map.values():
                rename_map[col] = COL_CHECK_ID
                continue
        if 'clear' in c_lower and 'date' in c_lower:
            rename_map[col] = COL_CLEAR_DATE
        elif 'post' in c_lower or 'txn' in c_lower or c_lower == 'date':
            rename_map[col] = COL_POST_DATE

    df = df.rename(columns=rename_map)

    date_cols = [c for c in df.columns if c in [COL_POST_DATE, COL_CLEAR_DATE]]
    for dc in date_cols:
        df[dc] = pd.to_datetime(df[dc], dayfirst=True, errors='coerce')

    if COL_VENDOR_ID in df.columns:
        df[COL_VENDOR_ID] = df[COL_VENDOR_ID].astype(str).str.upper().str.strip(" -_.")

    if COL_VENDOR_ID not in df.columns:
        df[COL_VENDOR_ID] = "Unknown_Vendor"
        if not df.empty:
            st.warning(f"⚠️ Column Warning: Could not find Vendor/Reference column.")

    return df

# ==========================================
# 1. VISUALIZATION HELPERS
# ==========================================
def plot_snowball_interactive(engine, open_checks, run_date, custom_delay=0):
    """ Forecast Plot with Smoothed Scenario Lines and deterministic-clear removal """
    fig = go.Figure()
    dates = [run_date + timedelta(days=i) for i in range(14)]

    base_cash = []
    hover_texts = []

    # Track checks that the model predicted to be deterministic-clears (effectively 100%) on some day
    predicted_cleared_ids = set()
    PRED_DET_EPS = 1.0 - 1e-9

    for d in dates:
        day_total = 0
        contributors = []
        for _, row in open_checks.iterrows():
            cid = str(row.get(COL_CHECK_ID, ''))
            if cid in predicted_cleared_ids:
                # once predicted 100%, exclude from future days
                continue

            # engine.predict_check carries forward over non-business days internally
            # use prev_business_day(d) for current_date_override
            try:
                cdo = prev_business_day(d)
            except Exception:
                cdo = None

            prob = engine.predict_check(row, d, current_date_override=cdo)
            sim_age = business_days_between(row[COL_POST_DATE], d)

            expected = row[COL_AMOUNT] * prob
            if expected > 0.50:
                day_total += expected
                contributors.append((cid, row[COL_AMOUNT], sim_age, prob))

            # If model is certain this check clears today, mark removed for subsequent days
            if prob >= PRED_DET_EPS:
                predicted_cleared_ids.add(cid)

        base_cash.append(day_total)

        contributors.sort(key=lambda x: x[1], reverse=True)
        top_list = contributors[:20]

        hover_html = f"<b>{d.strftime('%b-%d')}</b><br>Expected: <b>${day_total:,.0f}</b><br><br>Checks Contributing:"
        for cid, amt, age, p in top_list:
            hover_html += f"<br>• #{cid} (${amt:,.0f}) | Age: {age}d | Prob: {p:.0%}"
        if len(contributors) > 20:
            hover_html += f"<br><i>...and {len(contributors)-20} more</i>"
        hover_texts.append(hover_html)

    # Base Case Bar
    fig.add_trace(go.Bar(
        x=[d.strftime('%b-%d') for d in dates],
        y=base_cash,
        name="Expected Cash Flow",
        marker_color='#F5B041', opacity=0.8,
        text=[f"${v/1000:.0f}k" if v > 1000 else "" for v in base_cash],
        textposition='outside',
        hovertext=hover_texts, hoverinfo="text"
    ))

    # Delay Scenario (also respects deterministic-clear removal)
    if custom_delay > 0:
        delay_cash = []
        # use a fresh removal set per scenario (so scenario is independent)
        scenario_removed = set()
        for d in dates:
            sim_date = d - timedelta(days=custom_delay)
            day_total = 0
            for _, row in open_checks.iterrows():
                cid = str(row.get(COL_CHECK_ID, ''))
                if cid in scenario_removed:
                    continue

                if sim_date < run_date:
                    prob = 0
                else:
                    try:
                        cdo = prev_business_day(run_date)
                    except Exception:
                        cdo = None
                    prob = engine.predict_check(
                        row,
                        sim_date,
                        current_date_override=cdo
                    )

                if prob >= PRED_DET_EPS:
                    scenario_removed.add(cid)

                day_total += (row[COL_AMOUNT] * prob)
            delay_cash.append(day_total)

        fig.add_trace(go.Scatter(
            x=[d.strftime('%b-%d') for d in dates],
            y=delay_cash,
            name=f"Scenario: Delay +{custom_delay} Days",
            mode='lines',
            line=dict(width=4, color='#27AE60', shape='spline', smoothing=1.3)
        ))

    fig.update_layout(
        title="<b>Cash Flow Forecast</b>",
        xaxis_title="Future Date",
        yaxis_title="Expected Outflow ($)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    return fig

def plot_vendor_history_legacy(model, vendor_name, ledger):
    """ Classic History Plot with Smoothed CDF """
    history = ledger[
        (ledger[COL_VENDOR_ID] == vendor_name) &
        (ledger[COL_STATUS] == STATUS_CLEARED)
    ].copy()

    if history.empty:
        return None

    # Days_Taken computed in business days
    history['Days_Taken'] = history.apply(
        lambda r: business_days_between(r[COL_POST_DATE], r[COL_CLEAR_DATE]) if pd.notna(r[COL_CLEAR_DATE]) else None,
        axis=1
    )
    history = history[history['Days_Taken'] >= 0]
    if history.empty:
        return None

    daily_stats = history.groupby('Days_Taken').agg({
        COL_AMOUNT: 'sum',
        COL_CHECK_ID: list,
        COL_VENDOR_ID: 'count'
    }).rename(columns={COL_VENDOR_ID: 'Count'}).reset_index()

    daily_stats['Probability'] = daily_stats['Count'] / daily_stats['Count'].sum()
    daily_stats['Label'] = daily_stats[COL_AMOUNT].apply(
        lambda x: f"${x/1000:.0f}k" if x >= 1000 else f"${int(x)}"
    )

    def build_hover(row):
        ids = sorted([str(x) for x in row[COL_CHECK_ID]])[:5]
        id_str = "<br>• " + "<br>• ".join(ids)
        return (
            f"<b>Day {row['Days_Taken']}</b><br>"
            f"Prob: {row['Probability']:.1%}<br>"
            f"Vol: <b>${row[COL_AMOUNT]:,.0f}</b><br>"
            f"IDs:{id_str}"
        )

    daily_stats['Hover_Text'] = daily_stats.apply(build_hover, axis=1)

    fig = go.Figure()

    # 1. Histogram Bars (empirical)
    fig.add_trace(go.Bar(
        x=daily_stats['Days_Taken'],
        y=daily_stats['Probability'],
        text=daily_stats['Label'],
        textposition='outside',
        marker_color='#E67E22',
        opacity=0.6,
        hovertext=daily_stats['Hover_Text'],
        hoverinfo="text",
        name="Hist. Probability"
    ))

    # 2. Smoothed CDF line from model
    x_range = np.arange(0, int(daily_stats['Days_Taken'].max()) + 5)
    y_raw = [model.cdf(x) for x in x_range]

    x_smooth, y_smooth = smooth_line_data(x_range, y_raw)

    fig.add_trace(go.Scatter(
        x=x_smooth,
        y=y_smooth,
        name="Cumulative (CDF)",
        mode='lines',
        line=dict(color='#154360', width=3, shape='spline'),
        yaxis="y2"
    ))

    fig.update_layout(
        title=f"<b>Payment Profile: {vendor_name}</b>",
        yaxis=dict(title="Probability", tickformat=".0%"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 1.1]
        ),
        template="plotly_white",
        height=450,
        showlegend=False
    )

    return fig

def plot_interactive_landscape(ledger, run_date):
    """ Colorful Jitter """
    open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()
    if open_checks.empty: return None
    open_checks['Age'] = open_checks[COL_POST_DATE].apply(lambda d: business_days_between(d, run_date))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=open_checks['Age'], y=open_checks[COL_AMOUNT], mode='markers',
        marker=dict(
            size=open_checks[COL_AMOUNT].apply(lambda x: np.log(x)*3 if x>0 else 5),
            color=open_checks['Age'], colorscale='Portland', showscale=True,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=open_checks[COL_VENDOR_ID],
        hovertemplate="<b>%{text}</b><br>$%{y:,.2f}<br>%{x} days old"
    ))
    fig.update_layout(
        title="<b>Outstanding Check Landscape</b><br><sup>Color = Age | Size = Amount</sup>",
        xaxis_title="Days Since Posted", yaxis_title="Amount ($)",
        template="plotly_white", height=450
    )
    return fig

def smooth_line_data(x, y, points=300):
    """
    Applies Cubic Spline Interpolation to smooth curves.
    Clips Y values to ensure they don't dip below 0 (for amounts) or exceed 1 (for probs).
    """
    if len(x) < 4: return x, y

    try:
        x_new = np.linspace(min(x), max(x), points)
        spl = make_interp_spline(x, y, k=3)
        y_new = spl(x_new)

        if max(y) <= 1.0:
            y_new = np.clip(y_new, 0, 1)
        else:
            y_new = np.clip(y_new, 0, None)

        return x_new, y_new
    except:
        return x, y

# -----------------------------------------
# METRICS HELPERS (RMSE and Timing_Mismatch on business days)
# -----------------------------------------
def compute_business_day_metrics(df):
    """
    Returns (rmse_pct, Timing_Mismatch_pct) computed ONLY on business days present in df['Date'].
    df expected to have 'Date', 'Predicted', 'Actual' columns.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    mask = df['Date'].apply(lambda d: is_business_day(d))
    bd = df.loc[mask]

    if bd.empty:
        return 0.0, 0.0

    # RMSE percentage (normalize by mean actual on business days)
    mse = ((bd['Predicted'] - bd['Actual']) ** 2).mean()
    rmse = np.sqrt(mse)
    mean_actual = bd['Actual'].mean()
    rmse_pct = (rmse / mean_actual * 100) if mean_actual > 0 else 0.0

    # Timing_Mismatch % = sum |P-A| / sum A (business days)
    abs_err = np.abs(bd['Predicted'] - bd['Actual']).sum()
    total_act = bd['Actual'].sum()
    Timing_Mismatch_pct = (abs_err / total_act * 100) if total_act > 0 else 0.0

    return rmse_pct, Timing_Mismatch_pct

# ==========================================
# 2. MAIN APP
# ==========================================
mode = st.sidebar.radio("Navigation", ["🚀 Forecast & Intelligence", "🧪 Backtest Lab"])

if mode == "🚀 Forecast & Intelligence":
    st.header("🚀 Daily Cash Forecast")

    # --- UPDATE LEDGER FEATURE ---
    with st.expander("🔄 Update Ledger History (Add Cleared Checks)", expanded=False):
        st.caption("Upload a file with recently cleared checks. The system will remove duplicates and add new ones to history.")
        update_file = st.file_uploader("Upload Cleared Checks File", type=['xlsx', 'csv'], key="update_upl")

        if update_file and st.button("Merge into History"):
            with st.spinner("Merging..."):
                try:
                    new_df = pd.read_excel(update_file) if update_file.name.endswith('.xlsx') else pd.read_csv(update_file)
                    new_df = smart_normalize_columns(new_df)

                    master_path = "data/master_ledger.xlsx"
                    if os.path.exists(master_path):
                        master_df = pd.read_excel(master_path)
                    else:
                        master_df = pd.DataFrame()

                    if not master_df.empty:
                        master_df = smart_normalize_columns(master_df)

                    if COL_CHECK_ID in new_df.columns and COL_CHECK_ID in master_df.columns:
                        existing_ids = set(master_df[COL_CHECK_ID].astype(str))
                        unique_new = new_df[~new_df[COL_CHECK_ID].astype(str).isin(existing_ids)]

                        if not unique_new.empty:
                            updated_master = pd.concat([master_df, unique_new], ignore_index=True)
                            updated_master.to_excel(master_path, index=False)
                            st.success(f"✅ Success! Added {len(unique_new)} new cleared checks to history.")
                        else:
                            st.info("ℹ️ No new checks found. All checks in file already exist in history.")
                    else:
                        new_df.to_excel(master_path, index=False)
                        st.success(f"✅ Created new Master Ledger with {len(new_df)} rows.")

                except Exception as e:
                    st.error(f"Error updating ledger: {e}")

    # --- FORECAST INPUTS ---
    with st.expander("📂 Run Daily Forecast", expanded=True):
        c1, c2 = st.columns(2)
        run_date = c1.date_input("Run Date", value=datetime.today().date())
        run_date_pd = pd.to_datetime(run_date)
        uncleared_file = c2.file_uploader("Outstanding Checks (Required)", type=['xlsx', 'csv'])

        if c2.button("RUN FORECAST", type="primary"):
            if uncleared_file:
                with st.spinner("Processing..."):
                    # Save incoming file snapshot
                    os.makedirs(f"data/raw/{run_date_pd.strftime('%Y-%m-%d')}", exist_ok=True)
                    with open(f"data/raw/{run_date_pd.strftime('%Y-%m-%d')}/uncleared_checks.xlsx", "wb") as f:
                        f.write(uncleared_file.getbuffer())

                    # ingest + reconcile
                    ledger = ingest_and_reconcile(str(run_date_pd.strftime('%Y-%m-%d')), run_date_pd)

                    # attempt to normalize vendor col if missing
                    if COL_VENDOR_ID not in ledger.columns or (ledger[COL_VENDOR_ID] == 'Unknown_Vendor').all():
                        ledger = smart_normalize_columns(ledger)

                    st.session_state['ledger'] = ledger
                    engine = ForecastEngine(ledger)

                    open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()

                    # business-day handling: if run_date is non-business, carry predictions to next business day
                    if is_business_day(run_date_pd):
                        effective_date = run_date_pd
                        next_bd = None
                    else:
                        effective_date = None
                        next_bd = next_business_day(run_date_pd)

                    data = []
                    if effective_date is not None:
                        try:
                            cdo_effective = prev_business_day(effective_date)
                        except Exception:
                            cdo_effective = None
                    else:
                        cdo_effective = None

                    if next_bd is not None:
                        try:
                            cdo_next = prev_business_day(next_bd)
                        except Exception:
                            cdo_next = None
                    else:
                        cdo_next = None

                    predicted_today_total = 0.0
                    predicted_next_total = 0.0

                    for _, row in open_checks.iterrows():
                        # compute for effective date (today) if business day
                        if effective_date is not None:
                            prob = engine.predict_check(row, effective_date, current_date_override=cdo_effective)
                            expected = row[COL_AMOUNT] * prob
                            predicted_today_total += expected
                            row_out = row.to_dict()
                            row_out.update({'Probability': prob, 'Expected_Cash': expected, 'Effective_Date': effective_date.strftime('%Y-%m-%d')})
                            data.append(row_out)
                        else:
                            # today is non-business: predicted 0 today; still list rows and show they'd carry forward
                            row_out = row.to_dict()
                            row_out.update({'Probability': 0.0, 'Expected_Cash': 0.0, 'Effective_Date': next_bd.strftime('%Y-%m-%d') if next_bd is not None else ''})
                            data.append(row_out)

                        # compute for next business day (carry-forward)
                        if next_bd is not None:
                            prob_next = engine.predict_check(row, next_bd, current_date_override=cdo_next)
                            expected_next = row[COL_AMOUNT] * prob_next
                            predicted_next_total += expected_next

                    st.session_state['forecast_df'] = pd.DataFrame(data)
                    if is_business_day(run_date_pd):
                        st.session_state['predicted_total'] = predicted_today_total
                        st.session_state['carried_to_next_bd'] = None
                    else:
                        st.session_state['predicted_total'] = 0.0
                        st.session_state['carried_to_next_bd'] = {
                            'next_business_day': next_bd.strftime('%Y-%m-%d') if next_bd is not None else None,
                            'predicted_next_business_day': predicted_next_total
                        }

                    st.success("Done!")

    if st.session_state['forecast_df'] is not None:
        df = st.session_state['forecast_df']
        ledger = st.session_state['ledger']

        st.markdown("### 📊 Portfolio Overview")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("CASH REQUIRED TODAY", f"${st.session_state.get('predicted_total', 0.0):,.2f}")
            st.metric("TOTAL EXPOSURE", f"${df[COL_AMOUNT].sum():,.2f}")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer) as writer: df.to_excel(writer, index=False)
            st.download_button("📥 Download Excel", buffer.getvalue(), f"forecast_{run_date_pd.strftime('%Y-%m-%d')}.xlsx")
        with c2:
            st.plotly_chart(plot_interactive_landscape(ledger, run_date_pd), use_container_width=True)

        st.markdown("---")
        st.subheader("🕵️ Vendor Intelligence")

        valid_vendors = sorted([v for v in df[COL_VENDOR_ID].unique() if v != "Unknown_Vendor"])
        if not valid_vendors: valid_vendors = ["Unknown_Vendor"]

        sel_vendor = st.selectbox("Select Vendor:", valid_vendors)

        engine = ForecastEngine(ledger)

        if sel_vendor:
            c1, c2 = st.columns(2)
            with c1:
                if sel_vendor in engine.models['SPECIFIC']:
                    st.plotly_chart(plot_vendor_history_legacy(engine.models['SPECIFIC'][sel_vendor], sel_vendor, ledger), use_container_width=True)
                else:
                    st.info("Insufficient history for this vendor.")
            with c2:
                delay_days = st.slider(f"Simulate Delay for {sel_vendor}", 0, 14, 2)
                v_open = ledger[(ledger[COL_STATUS]==STATUS_OPEN) & (ledger[COL_VENDOR_ID]==sel_vendor)].copy()
                if not v_open.empty:
                    st.plotly_chart(plot_snowball_interactive(engine, v_open, run_date_pd, custom_delay=delay_days), use_container_width=True)
                    st.info(f"**Insight:** Moving the slider shows how cash flow shifts if {sel_vendor} is delayed.")
                    with st.expander("See Raw Data"):
                        v_open['Age'] = v_open[COL_POST_DATE].apply(lambda d: business_days_between(d, run_date_pd))
                        st.dataframe(v_open[[COL_CHECK_ID, COL_AMOUNT, COL_POST_DATE, 'Age']].sort_values('Age', ascending=False), use_container_width=True)
                else:
                    st.success("No open checks.")

# ------------------------------------------
# TAB 2: BACKTEST LAB
# ------------------------------------------
elif mode == "🧪 Backtest Lab":
    st.title("🧪 Backtest Lab")

    with st.expander("⚙️ Configuration", expanded=True):
        c1, c2 = st.columns(2)
        start_dt = c1.date_input("Start Date", value=(datetime.today()-timedelta(days=90)).date())
        end_dt = c1.date_input("End Date", value=(datetime.today()-timedelta(days=1)).date())
        hist_file = c2.file_uploader("Upload Master History", type=['xlsx'])

    if hist_file and st.button("RUN GLOBAL BACKTEST", type="primary"):
        with st.spinner("Processing History..."):
            try:
                raw = pd.read_excel(hist_file)
                cols_str = "".join([str(c) for c in raw.columns])
                if "Unnamed" in cols_str or raw.shape[1] < 3:
                    raw = pd.read_excel(hist_file, header=1)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()

            clean = smart_normalize_columns(raw)
            if COL_POST_DATE in clean.columns:
                clean[COL_POST_DATE] = pd.to_datetime(clean[COL_POST_DATE], dayfirst=True, errors='coerce')
            if COL_CLEAR_DATE in clean.columns:
                clean[COL_CLEAR_DATE] = pd.to_datetime(clean[COL_CLEAR_DATE], dayfirst=True, errors='coerce')

            if COL_CLEAR_DATE in clean.columns and COL_POST_DATE in clean.columns:
                clean[COL_DAYS_TO_SETTLE] = clean.apply(
                    lambda r: business_days_between(r[COL_POST_DATE], r[COL_CLEAR_DATE]) if pd.notna(r[COL_CLEAR_DATE]) else None,
                    axis=1
                )
                mask_cleared = clean[COL_CLEAR_DATE].notna()
                clean.loc[mask_cleared, 'Status'] = 'CLEARED'

            st.session_state['bt_clean'] = clean
            st.session_state['bt_dates'] = (start_dt, end_dt)

            res = run_walk_forward_backtest(clean, str(start_dt), str(end_dt))
            st.session_state['bt_res'] = res
            st.success("Global Backtest Complete.")

    if st.session_state['bt_res'] is not None:
        res = st.session_state['bt_res']

        st.markdown("### 🌍 Global Results")

        # Metric toggle (RMSE or Timing_Mismatch)
        metric_choice = st.radio("Primary Metric (computed on business days)", ["RMSE", "Timing_Mismatch"], horizontal=True)

        # --- GLOBAL METRICS ---
        act = res['Actual'].sum()
        pred = res['Predicted'].sum()
        var = pred - act

        # Compute metrics only on business days
        rmse_pct, Timing_Mismatch_pct = compute_business_day_metrics(res)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Actual", f"${act:,.0f}")
        m2.metric("Predicted", f"${pred:,.0f}")
        # protect against division by zero when showing delta
        delta_pct = (var/act*100) if act != 0 else 0.0
        m3.metric("Variance", f"${var:,.0f}", delta=f"{delta_pct:.1f}%")
        if metric_choice == "RMSE":
            m4.metric("% RMSE ", f"{rmse_pct:.1f}%", help="Root Mean Squared Error Percentage (business days only).")
        else:
            m4.metric("Timing Mismatch % ", f"{Timing_Mismatch_pct:.1f}%", help="Normalized Absolute Error % (business days only).")

        st.plotly_chart(plot_backtest_results(res, "Global"), use_container_width=True)

        st.markdown("---")
        st.subheader("🔬 Vendor Drill-Down")
        clean_df = st.session_state.get('bt_clean')

        if clean_df is not None:
            vendors = sorted([v for v in clean_df[COL_VENDOR_ID].astype(str).unique() if v != "Unknown_Vendor"])
            if not vendors: vendors = ["Unknown_Vendor"]

            c_sel, c_btn = st.columns([3, 1])
            v_sel = c_sel.selectbox("Select Vendor:", vendors)

            if c_btn.button(f"Run for {v_sel}"):
                s, e = st.session_state['bt_dates']
                with st.spinner(f"Simulating {v_sel}..."):
                    v_res = run_walk_forward_backtest(clean_df, str(s), str(e), vendor_filter=v_sel)
                    if not v_res.empty:
                        # compute totals for vendor
                        v_act = v_res['Actual'].sum()
                        v_pred = v_res['Predicted'].sum()
                        v_var = v_pred - v_act

                        # compute vendor metrics on business days
                        v_rmse_pct, v_Timing_Mismatch_pct = compute_business_day_metrics(v_res)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric(f"{v_sel} Actual", f"${v_act:,.0f}")
                        c2.metric(f"{v_sel} Predicted", f"${v_pred:,.0f}")
                        c3.metric("Variance", f"${v_var:,.0f}")
                        if metric_choice == "RMSE":
                            c4.metric("% RMSE ", f"{v_rmse_pct:.1f}%")
                        else:
                            c4.metric("Timing Mismatch % ", f"{v_Timing_Mismatch_pct:.1f}%")

                        st.plotly_chart(plot_backtest_results(v_res, v_sel), use_container_width=True)
                    else:
                        st.warning("No cleared checks found for this vendor in this period.")
