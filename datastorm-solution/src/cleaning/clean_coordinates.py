"""Clean outlet_coordinates.csv (Bronze → Silver).

Refactored version of the original src/poi/clean_coordinates.py, now using
the reusable DQ framework.

Checks applied:
  1. Null check on all columns.
  2. Duplicate check on Outlet_ID.
  3. Swapped lat/lon auto-correction (Sri Lanka: lat 5-10, lon 79-82).
  4. Range check on Latitude (5.5-10.2) and Longitude (79.0-82.5).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dq_checks import (
    check_duplicates,
    check_nulls,
    check_value_range,
    run_checks,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW = _PROJECT_ROOT / "data" / "bronze" / "raw"
_SILVER = _PROJECT_ROOT / "data" / "silver"
_REJECTED = _SILVER / "rejected_records"

COORDINATES_RAW = _RAW / "outlet_coordinates.csv"
COORDINATES_CLEAN = _SILVER / "outlet_coordinates_clean.csv"
COORDINATES_REJECTED = _REJECTED / "coordinates_rejected.csv"


def clean_coordinates() -> dict:
    """Clean the outlet coordinates dataset and return a summary dict."""
    _SILVER.mkdir(parents=True, exist_ok=True)
    _REJECTED.mkdir(parents=True, exist_ok=True)

    print("  Loading outlet coordinates...")
    df = pd.read_csv(COORDINATES_RAW)
    initial_rows = len(df)

    # --- Preserve original values for audit trail -------------------------
    df["original_latitude"] = df["Latitude"]
    df["original_longitude"] = df["Longitude"]

    # --- Auto-fix swapped coordinates -------------------------------------
    # Sri Lanka: latitude ~5-10, longitude ~79-82.
    # If lat > 20 and lon < 20, they're likely swapped.
    swap_mask = (df["Latitude"] > 20) & (df["Longitude"] < 20)
    swaps_fixed = swap_mask.sum()

    df.loc[swap_mask, ["Latitude", "Longitude"]] = (
        df.loc[swap_mask, ["Longitude", "Latitude"]].values
    )
    df["coordinate_swap_fixed"] = swap_mask

    # --- Run DQ checks (after swap fix) -----------------------------------
    checks = [
        (check_nulls, {
            "mandatory_cols": ["Outlet_ID", "Latitude", "Longitude"],
        }),
        (check_duplicates, {
            "key_cols": ["Outlet_ID"],
        }),
        (check_value_range, {
            "col": "Latitude", "min_val": 5.5, "max_val": 10.2,
        }),
        (check_value_range, {
            "col": "Longitude", "min_val": 79.0, "max_val": 82.5,
        }),
    ]

    clean_df, rejected_df = run_checks(df, checks)

    # --- Save outputs ------------------------------------------------------
    clean_df.to_csv(COORDINATES_CLEAN, index=False)
    rejected_df.to_csv(COORDINATES_REJECTED, index=False)

    summary = {
        "dataset": "outlet_coordinates.csv",
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "rejected_rows": len(rejected_df),
        "swaps_fixed": int(swaps_fixed),
        "rejection_breakdown": (
            rejected_df["check_name"].value_counts().to_dict()
            if not rejected_df.empty else {}
        ),
        "clean_path": str(COORDINATES_CLEAN),
        "rejected_path": str(COORDINATES_REJECTED),
    }

    print(f"  Coordinates cleaned: {summary['clean_rows']:,} / {summary['initial_rows']:,}")
    print(f"  Rejected: {summary['rejected_rows']:,}")
    print(f"  Swapped lat/lon fixed: {summary['swaps_fixed']}")

    return summary
