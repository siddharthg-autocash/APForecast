# src/apforecast/reporting/evolution.py
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from src.apforecast.core.constants import COL_AMOUNT, COL_VENDOR_ID, COL_CHECK_ID, COL_POST_DATE

def simulate_snowball_effect(engine, open_checks, start_date, simulation_days=3, horizon=14, save_plot=False, vendor_filter=None):
    """
    Generates Snowball Chart + Returns Data.
    Drill-Down Data is STRICTLY sorted by AGE (Oldest First).
    """
    
    if vendor_filter and vendor_filter != "ALL":
        open_checks = open_checks[open_checks[COL_VENDOR_ID] == vendor_filter].copy()
        title_suffix = f"for {vendor_filter}"
    else:
        title_suffix = "(All Vendors)"

    if open_checks.empty:
        return None, {}

    fig = go.Figure()
    LINE_COLORS = ['#154360', '#C0392B', '#27AE60', '#8E44AD'] 
    BAR_COLOR = '#E67E22'
    
    breakdown_data = {} 

    for i in range(simulation_days):
        sim_current_date = start_date + timedelta(days=i)
        scenario_label = f"Scenario: Delay {i} Day{'s' if i > 0 else ''}"
        breakdown_data[scenario_label] = {}
        
        daily_amounts = {}
        hover_texts = {} 
        
        plot_horizon = horizon + simulation_days 
        
        for d in range(plot_horizon):
            target_date = start_date + timedelta(days=d)
            if target_date < sim_current_date: continue
                
            day_contributors = []
            total_expected = 0
            
            for _, row in open_checks.iterrows():
                # Conditional Prob: P(Clear Today | Alive Yesterday)
                p_cum_target = engine.predict_check(row, target_date, current_date_override=sim_current_date)
                p_cum_yesterday = engine.predict_check(row, target_date - timedelta(days=1), current_date_override=sim_current_date)
                p_marginal = max(0, p_cum_target - p_cum_yesterday)
                
                expected_cash = row[COL_AMOUNT] * p_marginal
                
                if expected_cash > 0.01:
                    total_expected += expected_cash
                    age_at_payment = (target_date - row[COL_POST_DATE]).days
                    
                    day_contributors.append({
                        'Check_ID': row[COL_CHECK_ID],
                        'Vendor': row[COL_VENDOR_ID],
                        'Amount_Full': row[COL_AMOUNT],
                        'Expected_Cash': expected_cash,
                        'Age_Days': age_at_payment
                    })
            
            if total_expected > 0.01:
                daily_amounts[target_date] = total_expected
                
                # SORTING: Oldest Checks First
                df_contrib = pd.DataFrame(day_contributors)
                df_contrib = df_contrib.sort_values(by='Age_Days', ascending=False)
                breakdown_data[scenario_label][target_date] = df_contrib
                
                # Custom Hover
                limit = 20
                top_n = df_contrib.head(limit)
                hover_str = f"<b>Total: ${total_expected:,.0f}</b><br><br>Oldest Contributors:"
                for _, c in top_n.iterrows():
                    hover_str += f"<br>• #{c['Check_ID']} (Age: {c['Age_Days']}d) | ${c['Expected_Cash']:,.0f}"
                if len(df_contrib) > limit:
                    hover_str += f"<br><i>...and {len(df_contrib)-limit} more</i>"
                hover_texts[target_date] = hover_str

        dates = sorted(list(daily_amounts.keys()))
        values = [daily_amounts[d] for d in dates]
        hovers = [hover_texts[d] for d in dates]
        
        if i == 0:
            fig.add_trace(go.Bar(
                x=dates, y=values, name="Expected Cash Flow (Base Case)",
                marker_color=BAR_COLOR, opacity=0.5,
                hovertext=hovers, hoverinfo="text+x+name"
            ))

        fig.add_trace(go.Scatter(
            x=dates, y=values, mode='lines+markers', name=scenario_label,
            line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=3),
            marker=dict(size=6),
            hovertext=hovers, hoverinfo="text+x+name"
        ))

    fig.update_layout(
        title=dict(text=f"<b>Snowball Forecast: {title_suffix}</b>", font=dict(size=20)),
        xaxis_title="Future Date", yaxis_title="Expected Outflow ($)",
        yaxis_tickprefix="$", hovermode="closest", height=500
    )
    
    return fig, breakdown_data