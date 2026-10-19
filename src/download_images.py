
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from product_images import NOUN_QUERIES, product_noun  # noqa: E402

PRODUCTS_CSV = ROOT / "data" / "raw" / "products.csv"
OUT_CSV = ROOT / "assets" / "product_images.csv"
CACHE_JSON = ROOT / "assets" / "_photo_cache.json"
API_URL = "https://api.unsplash.com/search/photos"


class RateLimited(Exception):
    """Raised when Unsplash says we used up the hourly limit."""


def fetch_photos(query: str, access_key: str, n: int = 30) -> list:
    """Return a list of dicts: {image_url, photographer, page_url} for one search phrase."""
    resp = requests.get(
        API_URL,
        headers={"Authorization": f"Client-ID {access_key}"},
        params={"query": query, "per_page": n, "orientation": "landscape"},
        timeout=30,
    )
    if resp.status_code in (403, 429):
        raise RateLimited()
    resp.raise_for_status()
    photos = []
    for p in resp.json().get("results", []):
        photos.append({
            "image_url": p["urls"]["small"],            # about 400 px wide - good for cards
            "photographer": p.get("user", {}).get("name", ""),
            "page_url": p.get("links", {}).get("html", ""),
        })
    return photos


def load_cache() -> dict:
    if CACHE_JSON.exists():
        return json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CACHE_JSON.write_text(json.dumps(cache), encoding="utf-8")


def build_manifest(products: pd.DataFrame, fetch) -> list:
    """Give every product its own photo. `fetch(query)` returns the photo list."""
    products = products.copy()
    products["noun"] = [product_noun(t) for t in products["title"]]
    rows = []
    for noun, group in products.groupby("noun"):
        query = NOUN_QUERIES[noun]
        photos = fetch(query)
        if not photos:
            print(f"  [warning] no photos found for '{query}' - skipped")
            continue
        for i, pid in enumerate(sorted(group["product_id"])):
            photo = photos[i % len(photos)]      # each product gets the next photo
            rows.append({"product_id": int(pid), **photo})
    return rows


def main():
    access_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not access_key:
        sys.exit('UNSPLASH_ACCESS_KEY is not set. In PowerShell run:  $env:UNSPLASH_ACCESS_KEY = "your_key"')

    products = pd.read_csv(PRODUCTS_CSV)
    cache = load_cache()
    needed = sorted({NOUN_QUERIES[product_noun(t)] for t in products["title"]})
    todo = [q for q in needed if q not in cache]
    print(f"{len(needed)} product types, {len(needed) - len(todo)} already saved, {len(todo)} to fetch.")

    for q in todo:
        try:
            cache[q] = fetch_photos(q, access_key)
        except RateLimited:
            save_cache(cache)
            print(f"\nHourly limit reached. Progress saved ({len(cache)}/{len(needed)} types).")
            print("Wait about 1 hour, then run the same command again.")
            return
        print(f"  {q:<26} {len(cache[q])} photos")
        time.sleep(0.5)
    save_cache(cache)

    rows = build_manifest(products, lambda q: cache.get(q, []))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["product_id", "image_url", "photographer", "page_url"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. Saved {len(rows)} product photos to {OUT_CSV}")


if __name__ == "__main__":
    main()