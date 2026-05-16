"""Feature engineering stage for outlet/POI pairs."""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import (
    OUTLET_COORDINATES_CLEAN,
    POI_CLEAN_PATH,
    OUTLET_POI_FEATURES_PATH,
    GOLD_FEATURE_DIR,
    RADII,
)

EARTH_RADIUS_M = 6_371_000


def create_outlet_poi_features():
    GOLD_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    outlets = pd.read_csv(OUTLET_COORDINATES_CLEAN)
    pois = pd.read_csv(POI_CLEAN_PATH)

    outlet_coords_rad = np.radians(outlets[["Latitude", "Longitude"]].values)
    poi_coords_rad = np.radians(pois[["latitude", "longitude"]].values)

    tree = BallTree(poi_coords_rad, metric="haversine")

    features = outlets[["Outlet_ID"]].copy()

    poi_categories = sorted(pois["poi_category"].dropna().unique())

    for radius_name, radius_m in RADII.items():
        radius_rad = radius_m / EARTH_RADIUS_M
        nearby_indices = tree.query_radius(outlet_coords_rad, r=radius_rad)

        features[f"total_poi_count_{radius_name}"] = [
            len(indices) for indices in nearby_indices
        ]

        for category in poi_categories:
            category_counts = []

            for indices in nearby_indices:
                matched_pois = pois.iloc[indices]
                count = (matched_pois["poi_category"] == category).sum()
                category_counts.append(count)

            features[f"{category}_count_{radius_name}"] = category_counts

    for category in poi_categories:
        category_pois = pois[pois["poi_category"] == category]

        if category_pois.empty:
            features[f"nearest_{category}_m"] = np.nan
            continue

        category_coords_rad = np.radians(
            category_pois[["latitude", "longitude"]].values
        )

        category_tree = BallTree(category_coords_rad, metric="haversine")
        distances_rad, _ = category_tree.query(outlet_coords_rad, k=1)

        features[f"nearest_{category}_m"] = distances_rad[:, 0] * EARTH_RADIUS_M

    features["footfall_proxy_score"] = (
        3.0 * features.get("bus_stop_count_500m", 0)
        + 2.5 * features.get("school_count_500m", 0)
        + 2.0 * features.get("food_count_500m", 0)
        + 2.0 * features.get("commercial_count_500m", 0)
        + 1.5 * features.get("hospital_count_500m", 0)
        + 1.5 * features.get("tourism_count_1000m", 0)
    )

    max_score = features["footfall_proxy_score"].max()
    min_score = features["footfall_proxy_score"].min()

    if max_score > min_score:
        features["footfall_proxy_score_norm"] = (
            (features["footfall_proxy_score"] - min_score)
            / (max_score - min_score)
        )
    else:
        features["footfall_proxy_score_norm"] = 0

    features.to_csv(OUTLET_POI_FEATURES_PATH, index=False)

    print(f"POI features saved: {OUTLET_POI_FEATURES_PATH}")
    print(features.head())

    return features