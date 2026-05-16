"""Overpass API client helpers."""
import time
import requests
from .config import OVERPASS_URL, HEADERS


def fetch_overpass(query: str, retries: int = 5, sleep_seconds: int = 15):
    for attempt in range(retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers=HEADERS,
                timeout=120,
            )

            if response.status_code == 200:
                return response.json()

            print(f"Overpass status {response.status_code}. Retrying...")

        except requests.RequestException as error:
            print(f"Overpass request failed: {error}")

        time.sleep(sleep_seconds * (attempt + 1))

    return None