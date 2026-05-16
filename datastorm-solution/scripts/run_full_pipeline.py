"""Master pipeline: runs all 4 steps of the DataStorm 7.0 solution.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --step 1    # Run only Step 1
    python scripts/run_full_pipeline.py --step 1-3  # Run Steps 1 through 3
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def step1_clean_datasets():
    """Step 1: Clean all raw datasets (Bronze -> Silver)."""
    from src.cleaning.clean_transactions import clean_transactions
    from src.cleaning.clean_outlet_master import clean_outlet_master
    from src.cleaning.clean_coordinates import clean_coordinates
    from src.cleaning.clean_seasonality import clean_seasonality
    from src.cleaning.clean_holidays import clean_holidays

    summaries = []

    print("\n" + "=" * 60)
    print("STEP 1: Cleaning Raw Datasets (Bronze -> Silver)")
    print("=" * 60)

    print("\n[1/5] Cleaning outlet master...")
    summaries.append(clean_outlet_master())

    print("\n[2/5] Cleaning outlet coordinates...")
    summaries.append(clean_coordinates())

    print("\n[3/5] Cleaning distributor seasonality...")
    summaries.append(clean_seasonality())

    print("\n[4/5] Cleaning holidays...")
    summaries.append(clean_holidays())

    print("\n[5/5] Cleaning transactions (this may take a moment)...")
    summaries.append(clean_transactions())

    print("\n[OK] Step 1 complete: all datasets cleaned.")
    return summaries


def step2_generate_report(summaries):
    """Step 2: Generate the data cleaning report."""
    from src.cleaning.generate_report import generate_cleaning_report

    print("\n" + "=" * 60)
    print("STEP 2: Generating Cleaning Report")
    print("=" * 60)

    generate_cleaning_report(summaries)

    print("\n[OK] Step 2 complete: cleaning report generated.")


def step3_build_features():
    """Step 3: Build the final feature dataset (Silver -> Gold)."""
    from src.features.build_features import build_features

    print("\n" + "=" * 60)
    print("STEP 3: Building Final Feature Dataset (Silver -> Gold)")
    print("=" * 60)

    features = build_features()

    print(f"\n[OK] Step 3 complete: {features.shape[0]:,} outlets, {features.shape[1]} features.")
    return features


def step4_predict_potential():
    """Step 4: Run the potential prediction logic."""
    from src.prediction.predict_potential import predict_potential

    print("\n" + "=" * 60)
    print("STEP 4: Predicting Maximum Monthly Potential (Jan 2026)")
    print("=" * 60)

    predictions = predict_potential()

    print(f"\n[OK] Step 4 complete: {len(predictions):,} predictions generated.")
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="DataStorm 7.0 Full Pipeline -- Bronze -> Silver -> Gold -> Predictions"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="1-4",
        help="Step(s) to run. E.g. '1', '2', '1-3', '1-4' (default: 1-4)",
    )
    args = parser.parse_args()

    # Parse step range
    if "-" in args.step:
        start, end = map(int, args.step.split("-"))
    else:
        start = end = int(args.step)

    steps_to_run = set(range(start, end + 1))

    print("=" * 60)
    print("  DataStorm 7.0 -- Full Pipeline")
    print("  Team: 3-dots")
    print("=" * 60)
    print(f"\nRunning steps: {sorted(steps_to_run)}")

    t0 = time.time()
    summaries = None

    if 1 in steps_to_run:
        summaries = step1_clean_datasets()

    if 2 in steps_to_run:
        if summaries is None:
            print("\n[WARN] Step 2 requires Step 1 summaries. Running Step 1 first...")
            summaries = step1_clean_datasets()
        step2_generate_report(summaries)

    if 3 in steps_to_run:
        step3_build_features()

    if 4 in steps_to_run:
        step4_predict_potential()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed in {elapsed:.1f} seconds.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
