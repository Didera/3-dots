"""Predict Maximum Monthly Purchase Potential for January 2026.

Methodology: Quantile Ceiling + Peer Benchmarking + Constraint Adjustment

The core insight: historical volume is LEFT-CENSORED. It shows what outlets
DID sell, not what they COULD sell. Outlets may be constrained by:
  - Stockouts (insufficient supply)
  - Credit limits (can't order more)
  - Cooler capacity (can't store more)
  - Poor location awareness (untapped demand)

The potential is a THEORETICAL CEILING — we estimate it by:
  1. Segmenting outlets into peer groups (same type, size, region).
  2. Computing the P90/P95 ceiling of each peer group.
  3. Adjusting each outlet's potential based on its specific advantages
     (location quality via POI, infrastructure via coolers, growth trends).
  4. Applying January-specific seasonality adjustments.

Output: single aggregate per outlet → Outlet_ID + Maximum_Monthly_Liters
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLD = _PROJECT_ROOT / "data" / "gold" / "features"

FINAL_FEATURES = _GOLD / "final_features.csv"
PREDICTIONS_PATH = _GOLD / "3dots_predictions.csv"


def _assign_peer_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create peer groups by Outlet_Type and Outlet_Size.

    If a group has fewer than 10 members, merge it into a broader group
    (by Outlet_Type only) to ensure statistical robustness.
    """
    df = df.copy()

    # Primary grouping: Type + Size
    df["peer_group"] = df["Outlet_Type"].fillna("Unknown") + "_" + df["Outlet_Size"].fillna("Unknown")

    # Check group sizes and merge small groups
    group_sizes = df["peer_group"].value_counts()
    small_groups = group_sizes[group_sizes < 10].index

    for grp in small_groups:
        # Fall back to type-only grouping
        outlet_type = grp.split("_")[0]
        df.loc[df["peer_group"] == grp, "peer_group"] = outlet_type + "_ALL"

    return df


def _compute_peer_ceilings(df: pd.DataFrame, quantile: float = 0.90) -> pd.DataFrame:
    """Compute the volume ceiling for each peer group.

    The ceiling is defined as the given quantile of max_monthly_volume
    within the peer group. This represents what a well-performing outlet
    in this segment achieves.
    """
    ceilings = (
        df.groupby("peer_group")
        .agg(
            peer_ceiling=("max_monthly_volume", lambda x: x.quantile(quantile)),
            peer_median=("avg_monthly_volume", "median"),
            peer_p75=("max_monthly_volume", lambda x: x.quantile(0.75)),
            peer_count=("Outlet_ID", "count"),
            peer_avg_coolers=("Cooler_Count", "mean"),
            peer_avg_footfall=(
                "footfall_proxy_score_norm",
                lambda x: x.mean() if "footfall_proxy_score_norm" in df.columns else 0,
            ),
        )
        .reset_index()
    )

    return ceilings


def _calculate_location_factor(df: pd.DataFrame) -> pd.Series:
    """Score each outlet's location advantage based on POI features.

    High-footfall locations (near bus stops, schools, commercial areas)
    should have higher potential. Returns a factor in [0.8, 1.3].
    """
    if "footfall_proxy_score_norm" not in df.columns:
        return pd.Series(1.0, index=df.index)

    # Normalize to [0.8, 1.3] range — location can boost potential by up to 30%
    # or reduce it by up to 20%
    score = df["footfall_proxy_score_norm"].fillna(0)
    factor = 0.8 + 0.5 * score  # Range: [0.8, 1.3]

    return factor


def _calculate_infrastructure_factor(df: pd.DataFrame) -> pd.Series:
    """Score each outlet's infrastructure (cooler availability).

    More coolers → can store more product → higher potential.
    Returns a factor in [0.9, 1.2].
    """
    coolers = df["Cooler_Count"].fillna(0).clip(0, 5)
    # 0 coolers → 0.9x, 5 coolers → 1.2x
    factor = 0.9 + 0.06 * coolers

    return factor


def _calculate_growth_factor(df: pd.DataFrame) -> pd.Series:
    """Score each outlet's growth trajectory.

    Outlets trending upward are likely not yet at their ceiling.
    Returns a factor in [0.95, 1.15].
    """
    slope = df["volume_growth_slope"].fillna(0)

    # Normalize slope to [-1, 1] range using tanh-like capping
    max_abs = slope.abs().quantile(0.95) if not slope.empty else 1.0
    if max_abs == 0:
        max_abs = 1.0
    normalized = (slope / max_abs).clip(-1, 1)

    # Map to [0.95, 1.15] — growing outlets get a boost
    factor = 1.05 + 0.10 * normalized

    return factor


def _calculate_seasonality_factor(df: pd.DataFrame) -> pd.Series:
    """Apply January-specific seasonality adjustment.

    Returns a factor based on the distributor's January seasonality.
    """
    if "jan_seasonality_score" not in df.columns:
        return pd.Series(1.0, index=df.index)

    score = df["jan_seasonality_score"].fillna(0)
    # Favorable=1 → 1.10, Moderate=0 → 1.00, Un-Favorable=-1 → 0.90
    factor = 1.0 + 0.10 * score

    return factor


def _calculate_consistency_factor(df: pd.DataFrame) -> pd.Series:
    """Outlets with low CV (consistent sales) are likely close to their
    true potential already. Outlets with high CV have more upside.

    Returns a factor in [1.0, 1.2].
    """
    cv = df["volume_cv"].fillna(0).clip(0, 3)
    # High CV → more room for improvement → higher uplift factor
    factor = 1.0 + 0.067 * cv  # CV=3 → 1.2x

    return factor


def predict_potential() -> pd.DataFrame:
    """Run the full potential prediction pipeline."""
    _GOLD.mkdir(parents=True, exist_ok=True)

    print("  Loading final features...")
    df = pd.read_csv(FINAL_FEATURES)
    print(f"  Loaded {len(df):,} outlets with {len(df.columns)} features")

    # --- Step 1: Assign peer groups ---
    print("  Assigning peer groups...")
    df = _assign_peer_groups(df)
    n_groups = df["peer_group"].nunique()
    print(f"    -> {n_groups} peer groups created")

    # --- Step 2: Compute peer ceilings ---
    print("  Computing peer group ceilings (P90)...")
    ceilings = _compute_peer_ceilings(df, quantile=0.90)
    df = df.merge(ceilings, on="peer_group", how="left")

    # --- Step 3: Calculate adjustment factors ---
    print("  Calculating adjustment factors...")
    df["location_factor"] = _calculate_location_factor(df)
    df["infrastructure_factor"] = _calculate_infrastructure_factor(df)
    df["growth_factor"] = _calculate_growth_factor(df)
    df["seasonality_factor"] = _calculate_seasonality_factor(df)
    df["consistency_factor"] = _calculate_consistency_factor(df)

    # Combined adjustment
    df["combined_factor"] = (
        df["location_factor"]
        * df["infrastructure_factor"]
        * df["growth_factor"]
        * df["seasonality_factor"]
        * df["consistency_factor"]
    )

    # --- Step 4: Calculate final potential ---
    print("  Calculating potential ceiling...")

    # Method: Potential = max(historical_best, peer_ceiling * adjustments)
    # The potential should NEVER be less than the outlet's historical max

    # Base potential from peer ceiling
    df["peer_adjusted_potential"] = df["peer_ceiling"] * df["combined_factor"]

    # Historical best (using January-specific data if available)
    df["historical_best"] = df[["max_monthly_volume", "max_jan_volume"]].max(axis=1)

    # For outlets with very limited history, use peer median as floor
    low_data_mask = df["months_active"] < 3
    df.loc[low_data_mask, "historical_best"] = df.loc[
        low_data_mask, ["historical_best", "peer_median"]
    ].max(axis=1)

    # Final potential: the higher of historical best and peer-adjusted potential
    df["raw_potential"] = df[["historical_best", "peer_adjusted_potential"]].max(axis=1)

    # Apply January seasonality adjustment to final number
    df["Maximum_Monthly_Liters"] = df["raw_potential"] * df["seasonality_factor"]

    # Ensure minimum floor (no outlet should have zero potential if they have any history)
    min_floor = df.loc[df["total_volume"] > 0, "avg_monthly_volume"].quantile(0.05)
    df["Maximum_Monthly_Liters"] = df["Maximum_Monthly_Liters"].clip(lower=max(min_floor, 1.0))

    # Round to 2 decimal places
    df["Maximum_Monthly_Liters"] = df["Maximum_Monthly_Liters"].round(2)

    # --- Summary statistics ---
    print(f"\n  Prediction Summary:")
    print(f"    Outlets predicted:      {len(df):,}")
    print(f"    Mean potential:         {df['Maximum_Monthly_Liters'].mean():,.2f} L")
    print(f"    Median potential:       {df['Maximum_Monthly_Liters'].median():,.2f} L")
    print(f"    Min potential:          {df['Maximum_Monthly_Liters'].min():,.2f} L")
    print(f"    Max potential:          {df['Maximum_Monthly_Liters'].max():,.2f} L")
    print(f"    P25 potential:          {df['Maximum_Monthly_Liters'].quantile(0.25):,.2f} L")
    print(f"    P75 potential:          {df['Maximum_Monthly_Liters'].quantile(0.75):,.2f} L")

    # Uplift analysis
    df["uplift_pct"] = np.where(
        df["avg_monthly_volume"] > 0,
        (df["Maximum_Monthly_Liters"] / df["avg_monthly_volume"] - 1) * 100,
        0,
    )
    print(f"    Mean uplift vs avg:     {df['uplift_pct'].mean():,.1f}%")
    print(f"    Median uplift vs avg:   {df['uplift_pct'].median():,.1f}%")

    # --- Save predictions ---
    predictions = df[["Outlet_ID", "Maximum_Monthly_Liters"]].copy()
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\n  Predictions saved: {PREDICTIONS_PATH}")

    # Also save the detailed breakdown for analysis
    detail_path = _GOLD / "prediction_details.csv"
    detail_cols = [
        "Outlet_ID", "Outlet_Type", "Outlet_Size", "peer_group",
        "avg_monthly_volume", "max_monthly_volume", "historical_best",
        "peer_ceiling", "peer_median",
        "location_factor", "infrastructure_factor", "growth_factor",
        "seasonality_factor", "consistency_factor", "combined_factor",
        "peer_adjusted_potential", "raw_potential",
        "Maximum_Monthly_Liters", "uplift_pct",
    ]
    existing_cols = [c for c in detail_cols if c in df.columns]
    df[existing_cols].to_csv(detail_path, index=False)
    print(f"  Prediction details saved: {detail_path}")

    return predictions
