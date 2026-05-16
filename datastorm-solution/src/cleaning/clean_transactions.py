"""Clean transactions_history_final.csv (Bronze → Silver).

Checks applied:
  1. Null check on all mandatory columns.
  2. Duplicate check on composite key (Outlet_ID, Year, Month, Distributor_ID, SKU_ID).
  3. Range check on Year (2023-2025) and Month (1-12).
  4. Negative Volume_Liters are treated as returns — kept but flagged.
  5. Referential integrity: Outlet_ID → outlet_master, Distributor_ID → seasonality.
  6. Extreme-outlier flagging per SKU (IQR-based) — not rejected, but tagged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dq_checks import (
    check_duplicates,
    check_nulls,
    check_value_range,
    check_referential_integrity,
    run_checks,
)

# Paths (relative to project root)  ----------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW = _PROJECT_ROOT / "data" / "bronze" / "raw"
_SILVER = _PROJECT_ROOT / "data" / "silver"
_REJECTED = _SILVER / "rejected_records"

TRANSACTIONS_RAW = _RAW / "transactions_history_final.csv"
OUTLET_MASTER_RAW = _RAW / "outlet_master.csv"
SEASONALITY_RAW = _RAW / "distributor_seasonality_details.csv"

TRANSACTIONS_CLEAN = _SILVER / "transactions_clean.csv"
TRANSACTIONS_REJECTED = _REJECTED / "transactions_rejected.csv"


def clean_transactions() -> dict:
    """Clean the transactions dataset and return a summary dict."""
    _SILVER.mkdir(parents=True, exist_ok=True)
    _REJECTED.mkdir(parents=True, exist_ok=True)

    print("  Loading transactions...")
    df = pd.read_csv(TRANSACTIONS_RAW)
    initial_rows = len(df)

    # Load reference tables for referential integrity
    outlet_master = pd.read_csv(OUTLET_MASTER_RAW)
    seasonality = pd.read_csv(SEASONALITY_RAW)

    # --- Separate returns (negative volume) before DQ checks ---------------
    # Returns are valid business records — keep them but flag
    returns_mask = df["Volume_Liters"] < 0
    returns_count = returns_mask.sum()
    df["is_return"] = returns_mask

    # --- Run DQ checks sequentially ----------------------------------------
    checks = [
        (check_nulls, {
            "mandatory_cols": [
                "Outlet_ID", "Year", "Month",
                "Distributor_ID", "SKU_ID",
                "Volume_Liters", "Total_Bill_Value",
            ],
        }),
        (check_duplicates, {
            "key_cols": [
                "Outlet_ID", "Year", "Month",
                "Distributor_ID", "SKU_ID",
            ],
        }),
        (check_value_range, {"col": "Year", "min_val": 2023, "max_val": 2025}),
        (check_value_range, {"col": "Month", "min_val": 1, "max_val": 12}),
        (check_referential_integrity, {
            "col": "Outlet_ID",
            "ref_df": outlet_master,
            "ref_col": "Outlet_ID",
        }),
        (check_referential_integrity, {
            "col": "Distributor_ID",
            "ref_df": seasonality,
            "ref_col": "Distributor_ID",
        }),
    ]

    clean_df, rejected_df = run_checks(df, checks)

    # --- Outlier flagging (IQR per SKU) — NOT rejected, just tagged --------
    clean_df["is_volume_outlier"] = False
    for sku in clean_df["SKU_ID"].unique():
        sku_mask = clean_df["SKU_ID"] == sku
        # Only flag positive volumes (returns excluded from outlier detection)
        pos_mask = sku_mask & (clean_df["Volume_Liters"] > 0)
        vals = clean_df.loc[pos_mask, "Volume_Liters"]
        if vals.empty:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 3.0 * iqr  # Use 3x IQR for extreme outliers only
        outlier_mask = pos_mask & (clean_df["Volume_Liters"] > upper_fence)
        clean_df.loc[outlier_mask, "is_volume_outlier"] = True

    outlier_count = clean_df["is_volume_outlier"].sum()

    # --- Save outputs ------------------------------------------------------
    clean_df.to_csv(TRANSACTIONS_CLEAN, index=False)
    rejected_df.to_csv(TRANSACTIONS_REJECTED, index=False)

    summary = {
        "dataset": "transactions_history_final.csv",
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "rejected_rows": len(rejected_df),
        "returns_flagged": int(returns_count),
        "outliers_flagged": int(outlier_count),
        "rejection_breakdown": (
            rejected_df["check_name"].value_counts().to_dict()
            if not rejected_df.empty else {}
        ),
        "clean_path": str(TRANSACTIONS_CLEAN),
        "rejected_path": str(TRANSACTIONS_REJECTED),
    }

    print(f"  Transactions cleaned: {summary['clean_rows']:,} / {summary['initial_rows']:,}")
    print(f"  Rejected: {summary['rejected_rows']:,}")
    print(f"  Returns flagged: {summary['returns_flagged']:,}")
    print(f"  Outliers flagged: {summary['outliers_flagged']:,}")

    return summary
