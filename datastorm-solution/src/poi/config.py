"""Configuration for the POI pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "bronze" / "raw"
RAW_POI_DIR = PROJECT_ROOT / "data" / "bronze" / "poi_raw"

SILVER_DIR = PROJECT_ROOT / "data" / "silver"
SILVER_POI_DIR = SILVER_DIR / "poi_clean"

GOLD_FEATURE_DIR = PROJECT_ROOT / "data" / "gold" / "features"

OUTLET_COORDINATES_RAW = RAW_DATA_DIR / "outlet_coordinates.csv"
OUTLET_COORDINATES_CLEAN = SILVER_DIR / "outlet_coordinates_clean.csv"
REJECTED_COORDINATES = SILVER_DIR / "rejected_coordinates.csv"

OUTLET_MASTER_PATH = RAW_DATA_DIR / "outlet_master.csv"
TRANSACTIONS_PATH = RAW_DATA_DIR / "transactions_history_final.csv"

POI_CLEAN_PATH = SILVER_POI_DIR / "poi_clean.csv"
OUTLET_POI_FEATURES_PATH = GOLD_FEATURE_DIR / "outlet_poi_features.csv"
MODEL_READY_PATH = GOLD_FEATURE_DIR / "model_ready_with_poi.csv"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": "DataStorm-POI-Extraction/1.0 contact:dinithdevinda@gmail.com"
}

RADII = {
    "250m": 250,
    "500m": 500,
    "1000m": 1000,
}

POI_TAG_CONFIG = {
    "school": [
        ("amenity", "school"),
        ("amenity", "college"),
        ("amenity", "university"),
    ],
    "bus_stop": [
        ("highway", "bus_stop"),
        ("amenity", "bus_station"),
        ("public_transport", "station"),
    ],
    "railway_station": [
        ("railway", "station"),
        ("railway", "halt"),
    ],
    "hospital": [
        ("amenity", "hospital"),
        ("amenity", "clinic"),
        ("amenity", "pharmacy"),
    ],
    "food": [
        ("amenity", "restaurant"),
        ("amenity", "cafe"),
        ("amenity", "fast_food"),
    ],
    "commercial": [
        ("shop", "supermarket"),
        ("shop", "convenience"),
        ("amenity", "marketplace"),
    ],
    "tourism": [
        ("tourism", "attraction"),
        ("tourism", "hotel"),
        ("tourism", "guest_house"),
        ("tourism", "viewpoint"),
        ("tourism", "museum"),
    ],
    "religious": [
        ("amenity", "place_of_worship"),
    ],
    "fuel_station": [
        ("amenity", "fuel"),
    ],
}