"""POI extraction stage."""
import json
import time
import pandas as pd
from tqdm import tqdm

from .config import (
    OUTLET_COORDINATES_CLEAN,
    RAW_POI_DIR,
    SILVER_POI_DIR,
    POI_CLEAN_PATH,
    POI_TAG_CONFIG,
)
from .overpass_client import fetch_overpass


def create_bounding_boxes(df, step=0.20, buffer=0.02):
    min_lat = df["Latitude"].min() - buffer
    max_lat = df["Latitude"].max() + buffer
    min_lon = df["Longitude"].min() - buffer
    max_lon = df["Longitude"].max() + buffer

    boxes = []
    lat = min_lat

    while lat < max_lat:
        lon = min_lon

        while lon < max_lon:
            south = lat
            west = lon
            north = min(lat + step, max_lat)
            east = min(lon + step, max_lon)

            boxes.append((south, west, north, east))
            lon += step

        lat += step

    return boxes


def build_overpass_query(bbox):
    south, west, north, east = bbox

    query_parts = []

    for _, tag_pairs in POI_TAG_CONFIG.items():
        for key, value in tag_pairs:
            query_parts.append(f'node["{key}"="{value}"]({south},{west},{north},{east});')
            query_parts.append(f'way["{key}"="{value}"]({south},{west},{north},{east});')
            query_parts.append(f'relation["{key}"="{value}"]({south},{west},{north},{east});')

    query = f"""
    [out:json][timeout:90];
    (
      {' '.join(query_parts)}
    );
    out center tags;
    """

    return query


def classify_poi(tags):
    for category, tag_pairs in POI_TAG_CONFIG.items():
        for key, value in tag_pairs:
            if tags.get(key) == value:
                return category

    return "other"


def parse_poi_elements(elements):
    records = []

    for element in elements:
        tags = element.get("tags", {})

        lat = element.get("lat")
        lon = element.get("lon")

        if lat is None or lon is None:
            center = element.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")

        if lat is None or lon is None:
            continue

        records.append({
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "poi_name": tags.get("name"),
            "poi_category": classify_poi(tags),
            "latitude": lat,
            "longitude": lon,
            "raw_tags": json.dumps(tags, ensure_ascii=False),
        })

    return records


def extract_pois(test_mode=False):
    RAW_POI_DIR.mkdir(parents=True, exist_ok=True)
    SILVER_POI_DIR.mkdir(parents=True, exist_ok=True)

    outlets = pd.read_csv(OUTLET_COORDINATES_CLEAN)

    if test_mode:
        outlets = outlets.head(100)

    boxes = create_bounding_boxes(outlets)

    print(f"Bounding boxes generated: {len(boxes)}")

    all_records = []

    for i, bbox in enumerate(tqdm(boxes)):
        query = build_overpass_query(bbox)
        result = fetch_overpass(query)

        if result is None:
            print(f"Failed bbox {i}: {bbox}")
            continue

        raw_path = RAW_POI_DIR / f"overpass_bbox_{i:03d}.json"

        with open(raw_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False)

        records = parse_poi_elements(result.get("elements", []))
        all_records.extend(records)

        time.sleep(8)

    poi_df = pd.DataFrame(all_records)

    if poi_df.empty:
        print("No POIs collected.")
        return poi_df

    poi_df = poi_df.drop_duplicates(subset=["osm_type", "osm_id"])

    poi_df = poi_df[
        poi_df["latitude"].between(5.5, 10.2)
        & poi_df["longitude"].between(79.0, 82.5)
    ]

    poi_df.to_csv(POI_CLEAN_PATH, index=False)

    print(f"Clean POIs saved: {len(poi_df)}")
    print(poi_df["poi_category"].value_counts())

    return poi_df