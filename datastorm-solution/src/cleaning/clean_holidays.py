"""Clean holiday_list.csv (Bronze → Silver).

Checks applied:
  1. Null check on all columns.
  2. Duplicate check on (Date, Holiday_Name).
  3. Format check on Date (ISO 8601 datetime).
  4. Category check on Holiday_Type.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dq_checks import (
    check_duplicates,
    check_nulls,
    check_valid_categories,
    run_checks,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW = _PROJECT_ROOT / "data" / "bronze" / "raw"
_SILVER = _PROJECT_ROOT / "data" / "silver"
_REJECTED = _SILVER / "rejected_records"

HOLIDAYS_RAW = _RAW / "holiday_list.csv"
HOLIDAYS_CLEAN = _SILVER / "holidays_clean.csv"
HOLIDAYS_REJECTED = _REJECTED / "holidays_rejected.csv"

VALID_HOLIDAY_TYPES = {"Public", "Poya Day", "Bank", "Mercantile"}


def clean_holidays() -> dict:
    """Clean the holiday list dataset and return a summary dict."""
    _SILVER.mkdir(parents=True, exist_ok=True)
    _REJECTED.mkdir(parents=True, exist_ok=True)

    print("  Loading holiday list...")
    df = pd.read_csv(HOLIDAYS_RAW)
    initial_rows = len(df)

    # --- Parse dates -------------------------------------------------------
    df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date_parsed"].isnull().sum()

    # Strip whitespace
    df["Holiday_Type"] = df["Holiday_Type"].astype(str).str.strip()
    df["Holiday_Name"] = df["Holiday_Name"].astype(str).str.strip()

    # --- DQ checks ---------------------------------------------------------
    checks = [
        (check_nulls, {
            "mandatory_cols": ["Date", "Holiday_Name", "Holiday_Type"],
        }),
        (check_duplicates, {
            "key_cols": ["Date", "Holiday_Name", "Holiday_Type"],
        }),
        (check_valid_categories, {
            "col": "Holiday_Type",
            "valid_set": VALID_HOLIDAY_TYPES,
        }),
    ]

    clean_df, rejected_df = run_checks(df, checks)

    # Reject rows with unparseable dates
    if bad_dates > 0:
        bad_date_mask = clean_df["Date_parsed"].isnull()
        bad_date_rows = clean_df[bad_date_mask].copy()
        bad_date_rows["rejection_reason"] = "Unparseable date format"
        bad_date_rows["check_name"] = "format_check"
        rejected_df = pd.concat([rejected_df, bad_date_rows], ignore_index=True)
        clean_df = clean_df[~bad_date_mask].copy()

    # Normalize date to YYYY-MM-DD for consistency
    clean_df["Date"] = clean_df["Date_parsed"].dt.strftime("%Y-%m-%d")
    clean_df = clean_df.drop(columns=["Date_parsed"])

    # Also drop parsed date from rejected if it exists
    if "Date_parsed" in rejected_df.columns:
        rejected_df = rejected_df.drop(columns=["Date_parsed"])

    clean_df.to_csv(HOLIDAYS_CLEAN, index=False)
    rejected_df.to_csv(HOLIDAYS_REJECTED, index=False)

    summary = {
        "dataset": "holiday_list.csv",
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "rejected_rows": len(rejected_df),
        "bad_dates_found": int(bad_dates),
        "rejection_breakdown": (
            rejected_df["check_name"].value_counts().to_dict()
            if not rejected_df.empty else {}
        ),
        "clean_path": str(HOLIDAYS_CLEAN),
        "rejected_path": str(HOLIDAYS_REJECTED),
    }

    print(f"  Holidays cleaned: {summary['clean_rows']:,} / {summary['initial_rows']:,}")
    print(f"  Rejected: {summary['rejected_rows']:,}")

    return summary
