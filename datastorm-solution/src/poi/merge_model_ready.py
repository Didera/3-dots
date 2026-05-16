"""Merge step: combine transactions, outlet master, and POI features into a model-ready dataset."""
import pandas as pd

from .config import (
    TRANSACTIONS_PATH,
    OUTLET_MASTER_PATH,
    OUTLET_COORDINATES_CLEAN,
    OUTLET_POI_FEATURES_PATH,
    MODEL_READY_PATH,
    GOLD_FEATURE_DIR,
)


def create_model_ready_dataset():
    """Merge transaction history with outlet metadata, coordinates, and POI features.

    Join order:
      transactions  ->  outlet_master   (on Outlet_ID)
                    ->  coordinates     (on Outlet_ID)
                    ->  poi_features    (on Outlet_ID)

    Result is written to MODEL_READY_PATH.
    """
    GOLD_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load sources ---
    print("  Loading transactions...")
    transactions = pd.read_csv(TRANSACTIONS_PATH)

    print("  Loading outlet master...")
    outlet_master = pd.read_csv(OUTLET_MASTER_PATH)

    print("  Loading clean coordinates...")
    coordinates = pd.read_csv(OUTLET_COORDINATES_CLEAN)

    print("  Loading POI features...")
    poi_features = pd.read_csv(OUTLET_POI_FEATURES_PATH)

    # --- Merge ---
    print("  Merging datasets...")

    # Start with transactions as the base
    merged = transactions.merge(outlet_master, on="Outlet_ID", how="left")

    # Add coordinates (keep only lat/lon, drop helper columns)
    coord_cols = ["Outlet_ID", "Latitude", "Longitude"]
    merged = merged.merge(coordinates[coord_cols], on="Outlet_ID", how="left")

    # Add POI features
    merged = merged.merge(poi_features, on="Outlet_ID", how="left")

    # --- Summary ---
    print(f"  Transactions rows:      {len(transactions):,}")
    print(f"  Outlets with master:     {outlet_master['Outlet_ID'].nunique():,}")
    print(f"  Outlets with POI feats:  {poi_features['Outlet_ID'].nunique():,}")
    print(f"  Final merged rows:       {len(merged):,}")
    print(f"  Final merged columns:    {len(merged.columns)}")

    # Check for unmatched outlets
    null_poi = merged["footfall_proxy_score"].isna().sum()
    if null_poi > 0:
        print(f"  ⚠ {null_poi:,} rows missing POI features (outlets not in POI data)")

    null_master = merged["Outlet_Size"].isna().sum()
    if null_master > 0:
        print(f"  ⚠ {null_master:,} rows missing outlet master data")

    # --- Save ---
    merged.to_csv(MODEL_READY_PATH, index=False)
    print(f"  Model-ready dataset saved: {MODEL_READY_PATH}")

    return merged
