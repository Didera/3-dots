"""Build the final feature dataset (Silver → Gold).

Merges cleaned transactions, outlet master, coordinates, seasonality,
holidays, and POI features into one row-per-outlet feature table.

Feature groups:
  - Transaction aggregates (volume stats, bill value stats, growth)
  - Per-SKU volume shares
  - Outlet metadata (size, type, coolers)
  - Seasonality (January-specific)
  - Holiday features (January 2026)
  - POI features (from existing pipeline)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SILVER = _PROJECT_ROOT / "data" / "silver"
_GOLD = _PROJECT_ROOT / "data" / "gold" / "features"

# Silver inputs
TRANSACTIONS_CLEAN = _SILVER / "transactions_clean.csv"
OUTLET_MASTER_CLEAN = _SILVER / "outlet_master_clean.csv"
COORDINATES_CLEAN = _SILVER / "outlet_coordinates_clean.csv"
SEASONALITY_CLEAN = _SILVER / "seasonality_clean.csv"
HOLIDAYS_CLEAN = _SILVER / "holidays_clean.csv"

# Existing POI features (from POI pipeline)
POI_FEATURES = _GOLD / "outlet_poi_features.csv"

# Output
FINAL_FEATURES = _GOLD / "final_features.csv"


def _build_transaction_features(txn: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into per-outlet features."""

    # Exclude flagged returns for volume aggregation but keep for counting
    txn_positive = txn[txn["is_return"] == False].copy()

    # Monthly aggregation per outlet
    monthly = (
        txn_positive
        .groupby(["Outlet_ID", "Year", "Month"])
        .agg(
            monthly_volume=("Volume_Liters", "sum"),
            monthly_bill=("Total_Bill_Value", "sum"),
            monthly_txn_count=("Volume_Liters", "count"),
        )
        .reset_index()
    )

    # Per-outlet aggregation from monthly data
    outlet_agg = (
        monthly
        .groupby("Outlet_ID")
        .agg(
            total_volume=("monthly_volume", "sum"),
            total_bill_value=("monthly_bill", "sum"),
            avg_monthly_volume=("monthly_volume", "mean"),
            max_monthly_volume=("monthly_volume", "max"),
            min_monthly_volume=("monthly_volume", "min"),
            std_monthly_volume=("monthly_volume", "std"),
            median_monthly_volume=("monthly_volume", "median"),
            avg_monthly_bill=("monthly_bill", "mean"),
            months_active=("monthly_volume", "count"),
            total_transactions=("monthly_txn_count", "sum"),
        )
        .reset_index()
    )

    # Fill NaN std (outlets with only 1 month)
    outlet_agg["std_monthly_volume"] = outlet_agg["std_monthly_volume"].fillna(0)

    # Coefficient of variation
    outlet_agg["volume_cv"] = np.where(
        outlet_agg["avg_monthly_volume"] > 0,
        outlet_agg["std_monthly_volume"] / outlet_agg["avg_monthly_volume"],
        0,
    )

    # Growth trend (linear slope of monthly volume over time)
    monthly["time_index"] = (monthly["Year"] - 2023) * 12 + monthly["Month"]

    def _linear_slope(group):
        if len(group) < 2:
            return 0.0
        x = group["time_index"].values.astype(float)
        y = group["monthly_volume"].values.astype(float)
        x_mean, y_mean = x.mean(), y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    slopes = monthly.groupby("Outlet_ID").apply(_linear_slope).reset_index()
    slopes.columns = ["Outlet_ID", "volume_growth_slope"]
    outlet_agg = outlet_agg.merge(slopes, on="Outlet_ID", how="left")

    # Recency features
    monthly["month_seq"] = (monthly["Year"] - 2023) * 12 + monthly["Month"]

    recency = (
        monthly
        .groupby("Outlet_ID")
        .agg(
            first_active_month=("month_seq", "min"),
            last_active_month=("month_seq", "max"),
        )
        .reset_index()
    )
    # Jan 2026 = (2026 - 2023) * 12 + 1 = 37
    recency["months_since_first"] = 37 - recency["first_active_month"]
    recency["months_since_last"] = 37 - recency["last_active_month"]
    recency = recency.drop(columns=["first_active_month", "last_active_month"])

    outlet_agg = outlet_agg.merge(recency, on="Outlet_ID", how="left")

    # Per-SKU volume shares
    sku_volumes = (
        txn_positive
        .groupby(["Outlet_ID", "SKU_ID"])["Volume_Liters"]
        .sum()
        .reset_index()
    )
    sku_total = sku_volumes.groupby("Outlet_ID")["Volume_Liters"].sum().reset_index()
    sku_total.columns = ["Outlet_ID", "sku_total"]
    sku_volumes = sku_volumes.merge(sku_total, on="Outlet_ID")
    sku_volumes["share"] = sku_volumes["Volume_Liters"] / sku_volumes["sku_total"]

    sku_pivot = sku_volumes.pivot_table(
        index="Outlet_ID", columns="SKU_ID", values="share", fill_value=0,
    )
    sku_pivot.columns = [f"sku_share_{col}" for col in sku_pivot.columns]
    sku_pivot = sku_pivot.reset_index()

    outlet_agg = outlet_agg.merge(sku_pivot, on="Outlet_ID", how="left")

    # Distributor diversity
    dist_diversity = (
        txn_positive
        .groupby("Outlet_ID")
        .agg(
            num_distributors=("Distributor_ID", "nunique"),
            num_skus=("SKU_ID", "nunique"),
        )
        .reset_index()
    )
    outlet_agg = outlet_agg.merge(dist_diversity, on="Outlet_ID", how="left")

    # Returns count
    returns = (
        txn[txn["is_return"] == True]
        .groupby("Outlet_ID")
        .agg(return_count=("Volume_Liters", "count"),
             return_volume=("Volume_Liters", "sum"))
        .reset_index()
    )
    outlet_agg = outlet_agg.merge(returns, on="Outlet_ID", how="left")
    outlet_agg["return_count"] = outlet_agg["return_count"].fillna(0).astype(int)
    outlet_agg["return_volume"] = outlet_agg["return_volume"].fillna(0)

    # Historical January performance
    jan_data = monthly[(monthly["Month"] == 1)]
    if not jan_data.empty:
        jan_avg = (
            jan_data
            .groupby("Outlet_ID")
            .agg(avg_jan_volume=("monthly_volume", "mean"),
                 max_jan_volume=("monthly_volume", "max"))
            .reset_index()
        )
        outlet_agg = outlet_agg.merge(jan_avg, on="Outlet_ID", how="left")
        outlet_agg["avg_jan_volume"] = outlet_agg["avg_jan_volume"].fillna(0)
        outlet_agg["max_jan_volume"] = outlet_agg["max_jan_volume"].fillna(0)
    else:
        outlet_agg["avg_jan_volume"] = 0
        outlet_agg["max_jan_volume"] = 0

    # Outlier flag ratio
    outlier_ratio = (
        txn_positive
        .groupby("Outlet_ID")
        .agg(outlier_ratio=("is_volume_outlier", "mean"))
        .reset_index()
    )
    outlet_agg = outlet_agg.merge(outlier_ratio, on="Outlet_ID", how="left")
    outlet_agg["outlier_ratio"] = outlet_agg["outlier_ratio"].fillna(0)

    return outlet_agg


def _build_seasonality_features(seasonality: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    """Build seasonality features for January 2026 predictions."""

    # Get primary distributor per outlet (most frequent in transactions)
    txn_positive = txn[txn["is_return"] == False]
    primary_dist = (
        txn_positive
        .groupby("Outlet_ID")["Distributor_ID"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )
    primary_dist.columns = ["Outlet_ID", "primary_distributor"]

    # Get January seasonality for each distributor
    # Try Jan 2025 first, then Jan 2024, then Jan 2023 as fallback
    jan_season = seasonality[seasonality["Month"] == 1].copy()
    if not jan_season.empty:
        # Use the most recent January seasonality
        jan_season = jan_season.sort_values("Year", ascending=False)
        jan_season = jan_season.drop_duplicates(subset=["Distributor_ID"], keep="first")
        jan_season = jan_season[["Distributor_ID", "Seasonality_Index"]]
        jan_season.columns = ["primary_distributor", "jan_seasonality"]

        # Encode seasonality as numeric
        season_map = {"Un-Favorable": -1, "Moderate": 0, "Favorable": 1}
        jan_season["jan_seasonality_score"] = jan_season["jan_seasonality"].map(season_map)

        primary_dist = primary_dist.merge(jan_season, on="primary_distributor", how="left")
    else:
        primary_dist["jan_seasonality"] = "Unknown"
        primary_dist["jan_seasonality_score"] = 0

    primary_dist["jan_seasonality_score"] = primary_dist["jan_seasonality_score"].fillna(0)

    return primary_dist


def _build_holiday_features(holidays: pd.DataFrame) -> dict:
    """Count holidays in January 2026."""
    holidays["Date"] = pd.to_datetime(holidays["Date"])

    jan_2026 = holidays[
        (holidays["Date"].dt.year == 2026) & (holidays["Date"].dt.month == 1)
    ]

    # If no Jan 2026 holidays in data, use Jan 2025 as proxy
    if jan_2026.empty:
        jan_2026 = holidays[
            (holidays["Date"].dt.year == 2025) & (holidays["Date"].dt.month == 1)
        ]

    result = {
        "holidays_in_jan": len(jan_2026),
        "public_holidays_jan": len(jan_2026[jan_2026["Holiday_Type"] == "Public"]),
        "poya_days_jan": len(jan_2026[jan_2026["Holiday_Type"] == "Poya Day"]),
    }

    return result


def build_features() -> pd.DataFrame:
    """Build the final feature dataset and save to gold layer."""
    _GOLD.mkdir(parents=True, exist_ok=True)

    print("  Loading cleaned datasets...")
    txn = pd.read_csv(TRANSACTIONS_CLEAN)
    outlet_master = pd.read_csv(OUTLET_MASTER_CLEAN)
    coordinates = pd.read_csv(COORDINATES_CLEAN)
    seasonality = pd.read_csv(SEASONALITY_CLEAN)
    holidays = pd.read_csv(HOLIDAYS_CLEAN)

    # Load POI features if available
    poi_features = None
    if POI_FEATURES.exists():
        poi_features = pd.read_csv(POI_FEATURES)
        print(f"  POI features loaded: {poi_features.shape}")

    # --- Build feature groups ---
    print("  Building transaction features...")
    txn_features = _build_transaction_features(txn)
    print(f"    -> {txn_features.shape[1]} columns for {txn_features.shape[0]:,} outlets")

    print("  Building seasonality features...")
    season_features = _build_seasonality_features(seasonality, txn)

    print("  Building holiday features...")
    holiday_info = _build_holiday_features(holidays)

    # --- Merge all into final feature table ---
    print("  Merging feature groups...")

    # Start from outlet master (ensures we have all outlets)
    features = outlet_master.copy()

    # Add coordinates
    coord_cols = ["Outlet_ID", "Latitude", "Longitude"]
    features = features.merge(
        coordinates[coord_cols], on="Outlet_ID", how="left",
    )

    # Add transaction features
    features = features.merge(txn_features, on="Outlet_ID", how="left")

    # Add seasonality features
    features = features.merge(season_features, on="Outlet_ID", how="left")

    # Add holiday features (same for all outlets — broadcast)
    for key, val in holiday_info.items():
        features[key] = val

    # Add POI features
    if poi_features is not None:
        features = features.merge(poi_features, on="Outlet_ID", how="left")

    # --- Encode categoricals ---
    # Outlet_Size (ordinal)
    size_order = {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4, "Unknown": 0}
    features["outlet_size_encoded"] = features["Outlet_Size"].map(size_order).fillna(0)

    # Outlet_Type (one-hot)
    type_dummies = pd.get_dummies(features["Outlet_Type"], prefix="type")
    features = pd.concat([features, type_dummies], axis=1)

    # Jan seasonality (one-hot)
    if "jan_seasonality" in features.columns:
        season_dummies = pd.get_dummies(
            features["jan_seasonality"], prefix="seasonality"
        )
        features = pd.concat([features, season_dummies], axis=1)

    # --- Fill NaN for outlets with no transactions ---
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    # --- Save ---
    features.to_csv(FINAL_FEATURES, index=False)

    print(f"\n  Final feature dataset saved: {FINAL_FEATURES}")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {len(features.columns)}")

    return features
