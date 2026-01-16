# src/apforecast/reporting/dashboard.py

import pandas as pd
from datetime import timedelta


def build_forecast_dashboard(
    forecast_df: pd.DataFrame,
    run_date,
    horizon_days: int,
):
    """
    Aggregate expected outflow by future date.
    """

    rows = []

    for _, row in forecast_df.iterrows():
        for d in range(1, horizon_days + 1):
            rows.append({
                "forecast_date": run_date + timedelta(days=d),
                "expected_outflow": row["expected_outflow"] / horizon_days,
            })

    out_df = pd.DataFrame(rows)

    summary = (
        out_df
        .groupby("forecast_date", as_index=False)
        .sum()
        .sort_values("forecast_date")
    )

    return summary


def export_dashboard(df: pd.DataFrame, output_path):
    df.to_excel(output_path, index=False)
