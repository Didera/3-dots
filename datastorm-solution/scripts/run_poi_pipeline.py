"""CLI entrypoint for the POI pipeline."""
import argparse
from src.poi.pipeline import run_poi_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run POI pipeline using only first 100 outlets",
    )

    args = parser.parse_args()

    run_poi_pipeline(test_mode=args.test)


if __name__ == "__main__":
    main()