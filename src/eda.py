"""
eda.py
------
Exploratory Data Analysis: generates and saves all the charts requested
in the project spec into data/processed/eda_charts/, so they can be
embedded in the README / notebook / Streamlit analytics dashboard without
recomputation.

Charts produced:
    01_top_products.png            - most popular products (by weighted interactions)
    02_top_categories.png          - most popular categories
    03_rating_distribution.png     - distribution of ratings given
    04_interactions_per_user.png   - histogram of interaction counts per user
    05_interactions_per_product.png- histogram of interaction counts per product
    06_category_distribution.png   - product catalog composition by category
    07_price_distribution.png      - product price distribution
    08_user_activity_distribution.png - distribution of weighted user activity scores
    09_product_popularity_distribution.png - long-tail popularity curve
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils import PROCESSED_DIR, INTERACTION_WEIGHTS, get_logger

logger = get_logger("eda")
sns.set_theme(style="whitegrid")

CHART_DIR = PROCESSED_DIR / "eda_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {name}")


def run():
    products = pd.read_csv(PROCESSED_DIR / "products_clean.csv")
    interactions = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv", parse_dates=["timestamp"])
    interactions["weight"] = interactions["interaction_type"].map(INTERACTION_WEIGHTS)
    interactions = interactions.merge(products[["product_id", "category", "title"]],
                                       on="product_id", how="left")

    # 1. Most popular products
    top_products = (
        interactions.groupby("title")["weight"].sum().sort_values(ascending=False).head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_products.values, y=top_products.index, ax=ax, color="#4C72B0")
    ax.set_title("Top 15 Most Popular Products (weighted interaction score)")
    ax.set_xlabel("Weighted Interaction Score")
    ax.set_ylabel("")
    save(fig, "01_top_products.png")

    # 2. Most popular categories
    top_categories = interactions.groupby("category")["weight"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax, color="#55A868")
    ax.set_title("Category Popularity (weighted interaction score)")
    ax.set_xlabel("Weighted Interaction Score")
    save(fig, "02_top_categories.png")

    # 3. Rating distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=interactions.dropna(subset=["rating"])["rating"], ax=ax, color="#C44E52")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating (1-5)")
    save(fig, "03_rating_distribution.png")

    # 4. Interactions per user
    per_user = interactions.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(per_user, bins=40, ax=ax, color="#8172B2")
    ax.set_title("Number of Interactions per User")
    ax.set_xlabel("Interactions")
    save(fig, "04_interactions_per_user.png")

    # 5. Interactions per product
    per_product = interactions.groupby("product_id").size()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(per_product, bins=40, ax=ax, color="#CCB974")
    ax.set_title("Number of Interactions per Product")
    ax.set_xlabel("Interactions")
    save(fig, "05_interactions_per_product.png")

    # 6. Category distribution (catalog composition)
    cat_counts = products["category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Product Catalog Composition by Category")
    save(fig, "06_category_distribution.png")

    # 7. Price distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(products["price"], bins=40, kde=True, ax=ax, color="#64B5CD")
    ax.set_title("Product Price Distribution")
    ax.set_xlabel("Price ($)")
    save(fig, "07_price_distribution.png")

    # 8. User activity distribution (weighted)
    user_activity = interactions.groupby("user_id")["weight"].sum()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(user_activity, bins=40, ax=ax, color="#DD8452")
    ax.set_title("User Activity Score Distribution (weighted)")
    ax.set_xlabel("Activity Score")
    save(fig, "08_user_activity_distribution.png")

    # 9. Product popularity distribution (long tail)
    pop = interactions.groupby("product_id")["weight"].sum().sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(pop) + 1), pop.values, color="#937860")
    ax.set_title("Product Popularity Distribution (Long-Tail Curve)")
    ax.set_xlabel("Product Rank (most -> least popular)")
    ax.set_ylabel("Weighted Interaction Score")
    save(fig, "09_product_popularity_distribution.png")

    logger.info(f"All EDA charts saved to {CHART_DIR}")


if __name__ == "__main__":
    run()
