# src/apforecast/reporting/alerts.py

def generate_alerts(
    ledger_df,
    zombie_age_days: int = 45,
    large_amount_threshold: float = 50_000,
):
    """
    Generate operational alerts.
    """

    alerts = {}

    alerts["zombie_checks"] = ledger_df[
        (ledger_df["status"] == "OPEN")
        & (ledger_df["current_age_days"] >= zombie_age_days)
    ]

    alerts["large_checks"] = ledger_df[
        ledger_df["amount"] >= large_amount_threshold
    ]

    return alerts
