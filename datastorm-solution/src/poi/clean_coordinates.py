"""Coordinate cleaning stage for outlets."""
import pandas as pd
from .config import (
    OUTLET_COORDINATES_RAW,
    OUTLET_COORDINATES_CLEAN,
    REJECTED_COORDINATES,
    SILVER_DIR,
)


def clean_outlet_coordinates():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(OUTLET_COORDINATES_RAW)

    df["original_latitude"] = df["Latitude"]
    df["original_longitude"] = df["Longitude"]

    # Sri Lanka latitude is usually around 5-10.
    # Sri Lanka longitude is usually around 79-82.
    swap_mask = (df["Latitude"] > 20) & (df["Longitude"] < 20)

    df.loc[swap_mask, ["Latitude", "Longitude"]] = (
        df.loc[swap_mask, ["Longitude", "Latitude"]].values
    )

    df["coordinate_swap_fixed"] = swap_mask

    valid_mask = (
        df["Outlet_ID"].notna()
        & df["Latitude"].between(5.5, 10.2)
        & df["Longitude"].between(79.0, 82.5)
    )

    clean_df = df[valid_mask].copy()
    rejected_df = df[~valid_mask].copy()
    rejected_df["rejection_reason"] = "Invalid or missing Sri Lanka coordinate"

    clean_df.to_csv(OUTLET_COORDINATES_CLEAN, index=False)
    rejected_df.to_csv(REJECTED_COORDINATES, index=False)

    print(f"Clean coordinates: {len(clean_df)}")
    print(f"Rejected coordinates: {len(rejected_df)}")
    print(f"Swapped coordinates fixed: {swap_mask.sum()}")

    return clean_df