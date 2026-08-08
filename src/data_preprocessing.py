"""
data_preprocessing.py
----------------------
Loads the raw CSVs, cleans them, and saves analysis-ready processed data.

Cleaning steps performed (and why):
    1. Duplicate removal      -> exact duplicate interaction rows would
                                  double-count a single real event.
    2. Missing brand/description -> filled with explicit "Unknown" rather
                                  than dropped, because a missing brand is
                                  not a reason to discard an otherwise valid
                                  product; content-based features can still
                                  use the title/category/tags.
    3. Invalid ratings         -> ratings must be within 1-5. Anything
                                  outside that range (e.g. the synthetic
                                  "9" values injected by generate_dataset.py
                                  to mimic real-world data-entry errors) is
                                  clipped/removed since it cannot be a
                                  genuine rating.
    4. Orphan interactions     -> interactions referencing a user_id or
                                  product_id that doesn't exist in the
                                  users/products tables are dropped (would
                                  otherwise break joins downstream).
    5. Price sanity check      -> negative or zero prices are invalid for a
                                  retail catalog and are dropped.
    6. Type coercion           -> timestamps parsed to datetime, ids to int,
                                  so every downstream module can rely on
                                  consistent dtypes.

Run:
    python src/data_preprocessing.py
"""

import pandas as pd
import numpy as np

from utils import RAW_DIR, PROCESSED_DIR, get_logger

logger = get_logger("data_preprocessing")


def load_raw():
    products = pd.read_csv(RAW_DIR / "products.csv")
    users = pd.read_csv(RAW_DIR / "users.csv")
    interactions = pd.read_csv(RAW_DIR / "interactions.csv", parse_dates=["timestamp"])
    return products, users, interactions


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    before = len(products)
    products = products.drop_duplicates(subset="product_id").copy()

    products["brand"] = products["brand"].fillna("Unknown")
    products["description"] = products["description"].fillna(
        products["title"] + " - " + products["category"]
    )
    products["tags"] = products["tags"].fillna("")

    # invalid price sanity check
    invalid_price = products["price"] <= 0
    if invalid_price.any():
        logger.info(f"Dropping {invalid_price.sum()} products with non-positive price")
        products = products.loc[~invalid_price]

    products["price"] = products["price"].round(2)
    logger.info(f"Products: {before} -> {len(products)} after cleaning")
    return products.reset_index(drop=True)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    before = len(users)
    users = users.drop_duplicates(subset="user_id").copy()
    users["preferred_categories"] = users["preferred_categories"].fillna("")
    logger.info(f"Users: {before} -> {len(users)} after cleaning")
    return users.reset_index(drop=True)


def clean_interactions(interactions: pd.DataFrame, products: pd.DataFrame,
                        users: pd.DataFrame) -> pd.DataFrame:
    before = len(interactions)

    # 1. exact duplicate rows
    interactions = interactions.drop_duplicates().copy()

    # 2. drop orphan interactions (referencing ids that no longer exist)
    valid_products = set(products["product_id"])
    valid_users = set(users["user_id"])
    interactions = interactions[
        interactions["product_id"].isin(valid_products) &
        interactions["user_id"].isin(valid_users)
    ]

    # 3. unrealistic / invalid ratings: valid range is 1-5, everything else
    #    is treated as a data-entry error and removed rather than guessed.
    bad_rating_mask = interactions["rating"].notna() & (
        (interactions["rating"] < 1) | (interactions["rating"] > 5)
    )
    n_bad = bad_rating_mask.sum()
    if n_bad:
        logger.info(f"Found {n_bad} unrealistic ratings outside [1,5] -> setting to NaN")
        interactions.loc[bad_rating_mask, "rating"] = np.nan

    # 4. valid interaction types only
    valid_types = {"view", "add_to_cart", "wishlist", "purchase"}
    interactions = interactions[interactions["interaction_type"].isin(valid_types)]

    # 5. type coercion
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["product_id"] = interactions["product_id"].astype(int)

    interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Interactions: {before} -> {len(interactions)} after cleaning")
    return interactions


def run():
    products, users, interactions = load_raw()
    products = clean_products(products)
    users = clean_users(users)
    interactions = clean_interactions(interactions, products, users)

    products.to_csv(PROCESSED_DIR / "products_clean.csv", index=False)
    users.to_csv(PROCESSED_DIR / "users_clean.csv", index=False)
    interactions.to_csv(PROCESSED_DIR / "interactions_clean.csv", index=False)

    logger.info("Saved cleaned data to data/processed/")
    return products, users, interactions


if __name__ == "__main__":
    run()
