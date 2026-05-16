"""Clean distributor_seasonality_details.csv (Bronze → Silver).

Checks applied:
  1. Null check on all columns.
  2. Duplicate check on (Distributor_ID, Year, Month).
  3. Category check on Seasonality_Index.
  4. Range checks on Year (2023-2025) and Month (1-12).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dq_checks import (
    check_duplicates,
    check_nulls,
    check_valid_categories,
    check_value_range,
    run_checks,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW = _PROJECT_ROOT / "data" / "bronze" / "raw"
_SILVER = _PROJECT_ROOT / "data" / "silver"
_REJECTED = _SILVER / "rejected_records"

SEASONALITY_RAW = _RAW / "distributor_seasonality_details.csv"
SEASONALITY_CLEAN = _SILVER / "seasonality_clean.csv"
SEASONALITY_REJECTED = _REJECTED / "seasonality_rejected.csv"

VALID_SEASONALITY = {"Favorable", "Moderate", "Un-Favorable"}


def clean_seasonality() -> dict:
    """Clean the distributor seasonality dataset and return a summary dict."""
    _SILVER.mkdir(parents=True, exist_ok=True)
    _REJECTED.mkdir(parents=True, exist_ok=True)

    print("  Loading distributor seasonality...")
    df = pd.read_csv(SEASONALITY_RAW)
    initial_rows = len(df)

    # Strip whitespace from Seasonality_Index
    df["Seasonality_Index"] = df["Seasonality_Index"].astype(str).str.strip()

    checks = [
        (check_nulls, {
            "mandatory_cols": [
                "Distributor_ID", "Year", "Month", "Seasonality_Index",
            ],
        }),
        (check_duplicates, {
            "key_cols": ["Distributor_ID", "Year", "Month"],
        }),
        (check_value_range, {"col": "Year", "min_val": 2023, "max_val": 2025}),
        (check_value_range, {"col": "Month", "min_val": 1, "max_val": 12}),
        (check_valid_categories, {
            "col": "Seasonality_Index",
            "valid_set": VALID_SEASONALITY,
        }),
    ]

    clean_df, rejected_df = run_checks(df, checks)

    clean_df.to_csv(SEASONALITY_CLEAN, index=False)
    rejected_df.to_csv(SEASONALITY_REJECTED, index=False)

    summary = {
        "dataset": "distributor_seasonality_details.csv",
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "rejected_rows": len(rejected_df),
        "rejection_breakdown": (
            rejected_df["check_name"].value_counts().to_dict()
            if not rejected_df.empty else {}
        ),
        "clean_path": str(SEASONALITY_CLEAN),
        "rejected_path": str(SEASONALITY_REJECTED),
    }

    print(f"  Seasonality cleaned: {summary['clean_rows']:,} / {summary['initial_rows']:,}")
    print(f"  Rejected: {summary['rejected_rows']:,}")

    return summary
