# src/apforecast/main.py

import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from apforecast.ingestion.loader import get_raw_data_dir



from apforecast.core.dates import parse_run_date, age_in_days
from apforecast.core.constants import (
    MASTER_LEDGER_SCHEMA,
    FORECAST_HORIZON_DAYS,
)
from apforecast.core.config_loader import load_vendor_overrides
from apforecast.ingestion.reconciler import reconcile
from apforecast.modeling.engine import (
    build_probability_models,
    forecast_open_checks,
)
from apforecast.reporting.dashboard import (
    build_forecast_dashboard,
    export_dashboard,
)
from apforecast.reporting.alerts import generate_alerts


def main(run_date_str: str):
    run_date = parse_run_date(run_date_str)

    base = Path(__file__).resolve().parents[2]
    raw_dir = get_raw_data_dir(base, run_date_str)
    ledger_path = base / "data" / "processed" / "master_ledger.parquet"
    overrides_path = base / "data" / "config" / "vendor_strategy_overrides.xlsx"

    # -------------------------
    # Step 1: Reconciliation
    # -------------------------
    reconcile(run_date, raw_dir, ledger_path)

    # -------------------------
    # Load ledger
    # -------------------------
    ledger_df = pq.read_table(
        ledger_path,
        schema=MASTER_LEDGER_SCHEMA
    ).to_pandas()

    # compute age for OPEN checks
    ledger_df.loc[
        ledger_df["status"] == "OPEN",
        "current_age_days"
    ] = ledger_df.loc[
        ledger_df["status"] == "OPEN",
        "post_date"
    ].apply(lambda d: age_in_days(d, run_date))

    # -------------------------
    # Step 2: Modeling
    # -------------------------
    vendor_models, cohort_models = build_probability_models(ledger_df)
    override_rules = load_vendor_overrides(overrides_path)

    forecast_df = forecast_open_checks(
        ledger_df,
        vendor_models,
        cohort_models,
        override_rules,
        run_date,
        window=1,
    )

    # merge results back
    ledger_df.update(forecast_df)

    # -------------------------
    # Step 3: Reporting
    # -------------------------
    dashboard = build_forecast_dashboard(
        forecast_df,
        run_date,
        FORECAST_HORIZON_DAYS,
    )

    output_file = (
        base / "Forecast_Report_" + run_date_str + ".xlsx"
    )

    export_dashboard(dashboard, output_file)

    # -------------------------
    # Step 4: Alerts
    # -------------------------
    alerts = generate_alerts(ledger_df)

    print("Forecast generated:", output_file)
    print("Zombie checks:", len(alerts["zombie_checks"]))
    print("Large checks:", len(alerts["large_checks"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    main(args.date)
