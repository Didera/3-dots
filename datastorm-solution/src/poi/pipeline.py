"""Pipeline orchestration for POI enrichment."""
from .clean_coordinates import clean_outlet_coordinates
from .extract_pois import extract_pois
from .create_poi_features import create_outlet_poi_features
from .merge_model_ready import create_model_ready_dataset


def run_poi_pipeline(test_mode=False):
    print("Step 1: Cleaning outlet coordinates...")
    clean_outlet_coordinates()

    print("Step 2: Extracting POIs from Overpass...")
    poi_df = extract_pois(test_mode=test_mode)

    if poi_df is None or poi_df.empty:
        print("Skipping Steps 3-4: No POIs available for feature creation.")
        return

    print("Step 3: Creating outlet-level POI features...")
    create_outlet_poi_features()

    print("Step 4: Merging into model-ready dataset...")
    create_model_ready_dataset()

    print("POI pipeline completed.")