# src/apforecast/reporting/visuals.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.apforecast.core.constants import *

def plot_box_jitter_history(ledger, vendor_name):
    """
    Plots Historical Payment Behavior using a Box Plot + Jitter Points.
    - X Axis: Days Taken to Clear
    - Jitter Points: Individual Checks (Hover for details)
    """
    # Filter History
    history = ledger[
        (ledger[COL_VENDOR_ID] == vendor_name) & 
        (ledger['Status'] == 'CLEARED')
    ].copy()
    
    if history.empty:
        return None

    # Calculate Days
    history['Days_Taken'] = (history['Clear_Date'] - history[COL_POST_DATE]).dt.days
    history = history[history['Days_Taken'] >= 0]
    
    fig = go.Figure()

    # 1. Jitter Points (The Individual Checks)
    fig.add_trace(go.Box(
        x=history['Days_Taken'],
        name=vendor_name,
        boxpoints='all',          # Show all points
        jitter=0.5,               # Spread them out
        pointpos=-1.8,            # Place points under the box
        marker=dict(
            color='#E67E22',
            size=6,
            opacity=0.6
        ),
        line=dict(color='#2C3E50'),
        fillcolor='rgba(230, 126, 34, 0.2)', # Orange tint
        text=history.apply(
            lambda r: f"Check #{r[COL_CHECK_ID]}<br>${r[COL_AMOUNT]:,.2f}<br>Posted: {r[COL_POST_DATE].strftime('%Y-%m-%d')}", 
            axis=1
        ),
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"<b>Historical Behavior: {vendor_name}</b><br><sup>Distribution of Days Taken to Clear</sup>",
        xaxis_title="Days to Clear",
        yaxis_title="",
        showlegend=False,
        height=400,
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_forecast_distribution(open_checks, run_date):
    """
    Visualizes the Forecast Risks using Box + Jitter.
    - X Axis: Vendor Name (or 'Portfolio' if global)
    - Y Axis: Projected Days Outstanding (Age + Forecast)
    """
    # Calculate 'Current Age' for visualization context
    open_checks['Current_Age'] = (run_date - open_checks[COL_POST_DATE]).dt.days
    
    fig = go.Figure()
    
    # We plot the 'Current Age' distribution to show Risk Profile
    fig.add_trace(go.Box(
        y=open_checks['Current_Age'],
        name="Outstanding Portfolio",
        boxpoints='all',
        jitter=0.5,
        pointpos=-1.8,
        marker=dict(color='#3498DB', size=5, opacity=0.5),
        line=dict(color='#154360'),
        fillcolor='rgba(52, 152, 219, 0.2)',
        text=open_checks.apply(
            lambda r: f"<b>{r[COL_VENDOR_ID]}</b><br>Check #{r[COL_CHECK_ID]}<br>${r[COL_AMOUNT]:,.2f}<br>Age: {r['Current_Age']} days", 
            axis=1
        ),
        hoverinfo='text'
    ))

    fig.update_layout(
        title="<b>Outstanding Risk Profile</b><br><sup>Age Distribution of Unpaid Checks (Higher = Older/Riskier)</sup>",
        yaxis_title="Current Age (Days)",
        xaxis_title="",
        showlegend=False,
        height=500,
        template="plotly_white"
    )
    return fig