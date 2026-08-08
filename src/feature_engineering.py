"""
feature_engineering.py
------------------------
Builds all derived features used by the recommenders and the analytics
dashboard. Every feature is computed ONLY from data that would genuinely
be available at prediction time (no leakage from future interactions into
"historical" aggregates -- see build_train_test_split at the bottom).

Features created:
    Product-level:
        - avg_rating, n_ratings          : quality signal
        - weighted_interaction_score     : popularity using INTERACTION_WEIGHTS
        - popularity_rank                : percentile rank of the above
        - n_unique_users                 : how many distinct users engaged
        - price_bucket                   : Low / Medium / High / Premium
        - combined_text                  : title + brand + category + tags +
                                            description, used for TF-IDF

    User-level:
        - activity_score                 : weighted sum of all interactions
        - n_interactions                 : raw interaction count
        - avg_rating_given                : how generous/critical a rater is
        - top_category                    : category with most interactions
        - category_preference_vector      : normalized share of interactions
                                             per category (used for cold-start
                                             and explainability)

Why these and not others: each feature maps directly onto a piece of the
recommendation logic (content similarity needs `combined_text`; the
"popular in your category" cold-start path needs `popularity_rank` +
category; explanations need `top_category` / `avg_rating`).
"""

import pandas as pd
import numpy as np

from utils import PROCESSED_DIR, INTERACTION_WEIGHTS, get_logger

logger = get_logger("feature_engineering")


def build_product_features(products: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    products = products.copy()

    # --- rating aggregates ---
    rated = interactions.dropna(subset=["rating"])
    rating_agg = rated.groupby("product_id")["rating"].agg(["mean", "count"]).rename(
        columns={"mean": "avg_rating", "count": "n_ratings"}
    )
    products = products.merge(rating_agg, on="product_id", how="left")
    products["avg_rating"] = products["avg_rating"].fillna(0.0)
    products["n_ratings"] = products["n_ratings"].fillna(0).astype(int)

    # --- weighted popularity ---
    interactions = interactions.copy()
    interactions["weight"] = interactions["interaction_type"].map(INTERACTION_WEIGHTS)
    pop = interactions.groupby("product_id").agg(
        weighted_interaction_score=("weight", "sum"),
        n_unique_users=("user_id", "nunique"),
    )
    products = products.merge(pop, on="product_id", how="left")
    products["weighted_interaction_score"] = products["weighted_interaction_score"].fillna(0.0)
    products["n_unique_users"] = products["n_unique_users"].fillna(0).astype(int)

    products["popularity_rank"] = products["weighted_interaction_score"].rank(pct=True)

    # --- price bucket (quartile-based, so it adapts to the actual price
    #     distribution instead of hard-coded thresholds) ---
    try:
        products["price_bucket"] = pd.qcut(
            products["price"], q=4, labels=["Low", "Medium", "High", "Premium"]
        )
    except ValueError:
        products["price_bucket"] = "Medium"

    # --- combined text field for TF-IDF content-based similarity ---
    products["combined_text"] = (
        products["title"].fillna("") + " " +
        products["brand"].fillna("") + " " +
        products["category"].fillna("") + " " +
        products["tags"].fillna("").str.replace(",", " ", regex=False) + " " +
        products["description"].fillna("")
    ).str.lower()

    return products


def build_user_features(users: pd.DataFrame, interactions: pd.DataFrame,
                         products: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    interactions = interactions.merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )
    interactions["weight"] = interactions["interaction_type"].map(INTERACTION_WEIGHTS)

    activity = interactions.groupby("user_id").agg(
        activity_score=("weight", "sum"),
        n_interactions=("weight", "count"),
    )
    users = users.merge(activity, on="user_id", how="left")
    users["activity_score"] = users["activity_score"].fillna(0.0)
    users["n_interactions"] = users["n_interactions"].fillna(0).astype(int)

    rated = interactions.dropna(subset=["rating"])
    avg_rating_given = rated.groupby("user_id")["rating"].mean().rename("avg_rating_given")
    users = users.merge(avg_rating_given, on="user_id", how="left")
    users["avg_rating_given"] = users["avg_rating_given"].fillna(0.0)

    # top (most-interacted-with) category per user, weighted
    cat_weight = interactions.groupby(["user_id", "category"])["weight"].sum().reset_index()
    top_cat = (
        cat_weight.sort_values("weight", ascending=False)
        .drop_duplicates(subset="user_id")
        .set_index("user_id")["category"]
        .rename("top_category")
    )
    users = users.merge(top_cat, on="user_id", how="left")
    users["top_category"] = users["top_category"].fillna("Unknown")

    return users


def build_category_preference_matrix(interactions: pd.DataFrame,
                                      products: pd.DataFrame) -> pd.DataFrame:
    """Normalized per-user category preference share -> used for cold-start
    category recommendations and for the 'because you frequently buy from
    this category' explanation."""
    interactions = interactions.merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )
    interactions["weight"] = interactions["interaction_type"].map(INTERACTION_WEIGHTS)
    pivot = interactions.pivot_table(
        index="user_id", columns="category", values="weight", aggfunc="sum", fill_value=0.0
    )
    pref = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)
    return pref


def build_train_test_split(interactions: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Time-based split per user: the most recent `test_frac` of each
    user's interactions become the test set. This avoids leakage -- the
    model only ever trains on interactions that happened BEFORE the ones
    it's evaluated against, mirroring how a real system only knows the
    past when it recommends for "now"."""
    interactions = interactions.sort_values("timestamp")
    test_parts, train_parts = [], []
    for uid, grp in interactions.groupby("user_id"):
        n_test = max(1, int(len(grp) * test_frac)) if len(grp) >= 5 else 0
        if n_test == 0:
            train_parts.append(grp)
        else:
            train_parts.append(grp.iloc[:-n_test])
            test_parts.append(grp.iloc[-n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=interactions.columns)
    return train, test


def run():
    products = pd.read_csv(PROCESSED_DIR / "products_clean.csv")
    users = pd.read_csv(PROCESSED_DIR / "users_clean.csv")
    interactions = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv", parse_dates=["timestamp"])

    train, test = build_train_test_split(interactions)
    logger.info(f"Train/test split: {len(train)} train rows, {len(test)} test rows")

    product_features = build_product_features(products, train)
    user_features = build_user_features(users, train, products)
    category_pref = build_category_preference_matrix(train, products)

    product_features.to_csv(PROCESSED_DIR / "products_features.csv", index=False)
    user_features.to_csv(PROCESSED_DIR / "users_features.csv", index=False)
    category_pref.to_csv(PROCESSED_DIR / "category_preference_matrix.csv")
    train.to_csv(PROCESSED_DIR / "interactions_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "interactions_test.csv", index=False)

    logger.info("Saved engineered features to data/processed/")
    return product_features, user_features, train, test


if __name__ == "__main__":
    run()
