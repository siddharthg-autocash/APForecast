# src/apforecast/ingestion/reconciler.py

from pathlib import Path
from datetime import datetime, date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from apforecast.core.constants import MASTER_LEDGER_SCHEMA


# -----------------------------
# Helpers
# -----------------------------

def _parse_date(val):
    if pd.isna(val):
        return None
    return pd.to_datetime(val).date()


def _standardize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw bank / ERP columns into ledger-compatible names.
    """
    df = df.rename(columns={
        "Check #": "check_id",
        "Status": "status",
        "Type": "check_type",
        "Source": "source",
        "Post Date": "post_date",
        "Amount": "amount",
        "Reference": "reference",
        "BACS Reference": "bacs_reference",
        "Positive Pay": "positive_pay",
        "Void": "is_void",
        "Balanced": "is_balanced",
        "Cleared": "cleared_flag",
        "Cleared Date": "cleared_date",
    })

    # normalize dates
    df["post_date"] = df["post_date"].apply(_parse_date)
    if "cleared_date" in df:
        df["cleared_date"] = df["cleared_date"].apply(_parse_date)

    return df


def _empty_ledger() -> pa.Table:
    """
    Create an empty master ledger with the correct schema.
    """
    empty_df = pd.DataFrame({field.name: [] for field in MASTER_LEDGER_SCHEMA})
    return pa.Table.from_pandas(empty_df, schema=MASTER_LEDGER_SCHEMA)


# -----------------------------
# Main Reconciliation Logic
# -----------------------------

def reconcile(run_date: date, raw_dir: Path, ledger_path: Path):
    """
    Reconcile daily bank files into the master ledger.
    """

    # -------------------------
    # Load or initialize ledger
    # -------------------------

    if ledger_path.exists():
        ledger_tbl = pq.read_table(ledger_path, schema=MASTER_LEDGER_SCHEMA)
        ledger_df = ledger_tbl.to_pandas()
    else:
        ledger_df = _empty_ledger().to_pandas()

    # -------------------------
    # Load daily files
    # -------------------------

    cleared_fp = raw_dir / "bank_cleared.csv"
    issued_fp = raw_dir / "issued_checks.csv"

    cleared_df = _standardize_raw(pd.read_csv(cleared_fp))
    issued_df = _standardize_raw(pd.read_csv(issued_fp))

    # remove void checks early
    cleared_df = cleared_df[cleared_df["is_void"] != True]
    issued_df = issued_df[issued_df["is_void"] != True]

    # -------------------------
    # Process CLEARED checks
    # -------------------------

    for _, row in cleared_df.iterrows():
        check_id = row["check_id"]

        mask = ledger_df["check_id"] == check_id

        if mask.any():
            # transition OPEN -> CLEARED
            ledger_df.loc[mask, "status"] = "CLEARED"
            ledger_df.loc[mask, "cleared_flag"] = True
            ledger_df.loc[mask, "cleared_date"] = row["cleared_date"]
            ledger_df.loc[mask, "days_to_settle"] = (
                row["cleared_date"] - ledger_df.loc[mask, "post_date"].iloc[0]
            ).days
        else:
            # bootstrap cleared check
            new_row = {
                **row.to_dict(),
                "status": "CLEARED",
                "days_to_settle": (
                    row["cleared_date"] - row["post_date"]
                ).days,
            }
            ledger_df = pd.concat(
                [ledger_df, pd.DataFrame([new_row])],
                ignore_index=True,
            )

    # -------------------------
    # Process newly ISSUED checks
    # -------------------------

    existing_ids = set(ledger_df["check_id"])

    new_issued = issued_df[~issued_df["check_id"].isin(existing_ids)].copy()

    if not new_issued.empty:
        new_issued["status"] = "OPEN"
        new_issued["cleared_flag"] = False
        new_issued["cleared_date"] = None
        new_issued["days_to_settle"] = None

        ledger_df = pd.concat(
            [ledger_df, new_issued],
            ignore_index=True,
        )

    # -------------------------
    # Update audit fields
    # -------------------------

    ledger_df["last_updated_run"] = run_date

    # -------------------------
    # Enforce schema + persist
    # -------------------------

    ledger_tbl = pa.Table.from_pandas(
        ledger_df,
        schema=MASTER_LEDGER_SCHEMA,
        preserve_index=False,
    )

    pq.write_table(ledger_tbl, ledger_path)
