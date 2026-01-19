# src/apforecast/reporting/interactive.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.apforecast.core.constants import *

def plot_interactive_outstanding_analysis(open_checks, run_date):
    """
    Generates an Interactive Jitter + Box Plot.
    User can hover over individual dots to see Check Age & Details.
    """
    if open_checks.empty:
        return None

    # 1. Prepare Data
    df = open_checks.copy()
    
    # Calculate Age
    df['Age_Days'] = (run_date - df[COL_POST_DATE]).dt.days
    df['Age_Days'] = df['Age_Days'].apply(lambda x: max(0, x)) # No negative age
    
    # Create a truncated Vendor Name for cleaner X-axis
    df['Vendor_Short'] = df[COL_VENDOR_ID].astype(str).str.slice(0, 15)
    
    # Create Hover Text
    df['Hover_Info'] = (
        "<b>Vendor:</b> " + df[COL_VENDOR_ID].astype(str) + "<br>" +
        "<b>Check ID:</b> " + df[COL_CHECK_ID].astype(str) + "<br>" +
        "<b>Amount:</b> $" + df[COL_AMOUNT].apply(lambda x: "{:,.2f}".format(x)) + "<br>" +
        "<b>Age:</b> " + df['Age_Days'].astype(str) + " days old"
    )

    # 2. Build the Plot (Strip Plot = Jitter)
    # FIX: Removed 'color_continuous_scale' from here to prevent the error
    fig = px.strip(
        df, 
        x='Vendor_Short', 
        y=COL_AMOUNT, 
        color='Age_Days',
        custom_data=[COL_VENDOR_ID, COL_CHECK_ID, 'Age_Days', COL_AMOUNT],
        title=f"Outstanding Checks Analysis (Jitter + Box)<br><sup>Hover over dots to see specific check details</sup>"
    )
    
    # FIX: Apply the 'Red-Yellow-Green' color scale safely via layout
    fig.update_layout(coloraxis=dict(colorscale='RdYlGn_r'))

    # 3. Add Box Plot Layer (for Distribution stats)
    fig.add_trace(
        go.Box(
            y=df[COL_AMOUNT],
            x=df['Vendor_Short'],
            name="Distribution",
            boxpoints=False, # We already have points from px.strip
            fillcolor='rgba(0,0,0,0)', # Transparent fill
            line=dict(color='gray', width=1.5),
            hoverinfo='skip' # Don't clutter hover with box stats
        )
    )

    # 4. Styling
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Amount: $%{y:,.0f}<br>Age: %{customdata[2]} days<br>Check ID: %{customdata[1]}"
    )
    
    fig.update_layout(
        xaxis_title="Vendor",
        yaxis_title="Check Amount ($)",
        yaxis_tickprefix="$",
        height=600,
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=14)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')

    return fig