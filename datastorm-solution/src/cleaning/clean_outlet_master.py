"""Clean outlet_master.csv (Bronze → Silver).

Checks applied:
  1. Null check on mandatory columns (Outlet_ID, Outlet_Type, Cooler_Count).
  2. Duplicate check on Outlet_ID.
  3. Typo auto-correction for Outlet_Type and Outlet_Size.
  4. Range check on Cooler_Count (0-10).
  5. Null Outlet_Size imputed as 'Unknown'.
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

OUTLET_MASTER_RAW = _RAW / "outlet_master.csv"
OUTLET_MASTER_CLEAN = _SILVER / "outlet_master_clean.csv"
OUTLET_MASTER_REJECTED = _REJECTED / "outlet_master_rejected.csv"

# --- Typo correction maps -------------------------------------------------

OUTLET_TYPE_CORRECTIONS = {
    "Grocry": "Grocery",
    "Bakry": "Bakery",
    " Eatery ": "Eatery",
}

OUTLET_SIZE_CORRECTIONS = {
    "small": "Small",
}


def clean_outlet_master() -> dict:
    """Clean the outlet master dataset and return a summary dict."""
    _SILVER.mkdir(parents=True, exist_ok=True)
    _REJECTED.mkdir(parents=True, exist_ok=True)

    print("  Loading outlet master...")
    df = pd.read_csv(OUTLET_MASTER_RAW)
    initial_rows = len(df)

    # --- Typo corrections (applied BEFORE DQ checks) ----------------------
    typos_fixed_type = 0
    for wrong, right in OUTLET_TYPE_CORRECTIONS.items():
        mask = df["Outlet_Type"].str.strip() == wrong.strip()
        typos_fixed_type += mask.sum()
        df.loc[mask, "Outlet_Type"] = right

    # Also strip whitespace from Outlet_Type
    df["Outlet_Type"] = df["Outlet_Type"].str.strip()

    typos_fixed_size = 0
    for wrong, right in OUTLET_SIZE_CORRECTIONS.items():
        mask = df["Outlet_Size"] == wrong
        typos_fixed_size += mask.sum()
        df.loc[mask, "Outlet_Size"] = right

    # --- Impute null Outlet_Size as 'Unknown' ------------------------------
    null_size_count = df["Outlet_Size"].isnull().sum()
    df["Outlet_Size"] = df["Outlet_Size"].fillna("Unknown")

    # --- Run DQ checks -----------------------------------------------------
    checks = [
        (check_nulls, {
            "mandatory_cols": ["Outlet_ID", "Outlet_Type", "Cooler_Count"],
        }),
        (check_duplicates, {
            "key_cols": ["Outlet_ID"],
        }),
        (check_value_range, {
            "col": "Cooler_Count", "min_val": 0, "max_val": 10,
        }),
    ]

    clean_df, rejected_df = run_checks(df, checks)

    # --- Save outputs ------------------------------------------------------
    clean_df.to_csv(OUTLET_MASTER_CLEAN, index=False)
    rejected_df.to_csv(OUTLET_MASTER_REJECTED, index=False)

    summary = {
        "dataset": "outlet_master.csv",
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "rejected_rows": len(rejected_df),
        "typos_fixed_type": int(typos_fixed_type),
        "typos_fixed_size": int(typos_fixed_size),
        "null_size_imputed": int(null_size_count),
        "rejection_breakdown": (
            rejected_df["check_name"].value_counts().to_dict()
            if not rejected_df.empty else {}
        ),
        "clean_path": str(OUTLET_MASTER_CLEAN),
        "rejected_path": str(OUTLET_MASTER_REJECTED),
        "final_outlet_types": sorted(clean_df["Outlet_Type"].unique().tolist()),
        "final_outlet_sizes": sorted(clean_df["Outlet_Size"].unique().tolist()),
    }

    print(f"  Outlet master cleaned: {summary['clean_rows']:,} / {summary['initial_rows']:,}")
    print(f"  Rejected: {summary['rejected_rows']:,}")
    print(f"  Typos fixed (type): {summary['typos_fixed_type']}")
    print(f"  Typos fixed (size): {summary['typos_fixed_size']}")
    print(f"  Null Outlet_Size imputed: {summary['null_size_imputed']}")

    return summary
