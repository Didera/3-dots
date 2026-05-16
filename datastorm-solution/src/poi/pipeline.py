"""Pipeline orchestration for POI enrichment."""
from .clean_coordinates import clean_outlet_coordinates
from .extract_pois import extract_pois
from .create_poi_features import create_outlet_poi_features


def run_poi_pipeline(test_mode=False):
    print("Step 1: Cleaning outlet coordinates...")
    clean_outlet_coordinates()

    print("Step 2: Extracting POIs from Overpass...")
    extract_pois(test_mode=test_mode)

    print("Step 3: Creating outlet-level POI features...")
    create_outlet_poi_features()

    print("POI pipeline completed.")