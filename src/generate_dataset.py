"""
generate_dataset.py
--------------------
Generates a realistic, statistically-sound e-commerce dataset:
    - data/raw/products.csv       (product catalog)
    - data/raw/users.csv          (user profiles)
    - data/raw/interactions.csv   (user-product interaction log)

WHY A GENERATED DATASET?
This project was built inside a sandboxed environment with no internet
access, so a live download from Kaggle (e.g. "Amazon Product Reviews",
"Olist Brazilian E-Commerce", "H&M Personalized Recommendations") was not
possible. Instead of hard-coding a handful of rows, this script generates
a large, statistically realistic dataset that reproduces the properties
real e-commerce data is known to have:

    - Zipf-distributed product popularity (a few products get most of the
      interactions -- the classic "long tail").
    - Category-correlated user preferences (users mostly repeat-buy inside
      2-3 preferred categories, occasionally exploring others).
    - Realistic interaction-type funnel: view > add_to_cart > wishlist >
      purchase (fewer users convert at each stage).
    - Ratings that are only present for a subset of purchases, and are
      skewed toward 4-5 stars (a well-documented real-world bias).
    - Noisy, partially-missing metadata (some descriptions/brands missing)
      so the preprocessing step in data_preprocessing.py has real work to do.

Because this script and all downstream code use plain, standard CSV
columns (user_id, product_id, category, brand, price, rating,
interaction_type, timestamp ...), you can swap this synthetic data for a
real dataset at any time -- just make sure your CSVs use the same column
names, or edit the column-mapping constants at the top of
data_preprocessing.py.

Run:
    python src/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_USERS = 2000
N_PRODUCTS = 800
N_INTERACTIONS = 60000

CATEGORIES = {
    "Electronics": ["TechNova", "Sonik", "PixelPro", "Voltix"],
    "Fashion": ["UrbanThread", "Lunaire", "Northgear", "Velure"],
    "Home & Kitchen": ["HearthCo", "CozyNest", "PrimeHaus", "Kindle & Wick"],
    "Beauty & Personal Care": ["GlowLab", "PureSkin", "AromaBloom", "Verdant"],
    "Sports & Outdoors": ["PeakForm", "TrailBlaze", "IronCore", "SwiftStride"],
    "Books": ["Inkwell Press", "Nova Publishing", "Chapter House", "Quill & Co"],
    "Toys & Games": ["FunForge", "PlayNest", "Brickbound", "Wonderkit"],
    "Grocery": ["FreshField", "PantryPlus", "GreenHarvest", "DailyBasket"],
}
CATEGORY_LIST = list(CATEGORIES.keys())

ADJECTIVES = ["Premium", "Compact", "Wireless", "Eco-Friendly", "Classic",
              "Portable", "Deluxe", "Smart", "Everyday", "Pro", "Ultra",
              "Handcrafted", "Lightweight", "All-in-One", "Signature"]
NOUNS = {
    "Electronics": ["Headphones", "Speaker", "Smartwatch", "Charger", "Router",
                    "Earbuds", "Webcam", "Power Bank", "Keyboard", "Monitor"],
    "Fashion": ["Jacket", "Sneakers", "Backpack", "Sunglasses", "Scarf",
                "T-Shirt", "Handbag", "Wallet", "Belt", "Hat"],
    "Home & Kitchen": ["Blender", "Cookware Set", "Air Fryer", "Bedding Set",
                       "Vacuum Cleaner", "Coffee Maker", "Lamp", "Organizer"],
    "Beauty & Personal Care": ["Face Serum", "Shampoo", "Moisturizer",
                               "Lipstick", "Perfume", "Hair Dryer", "Razor"],
    "Sports & Outdoors": ["Yoga Mat", "Dumbbell Set", "Running Shoes",
                          "Camping Tent", "Water Bottle", "Bike Helmet"],
    "Books": ["Novel", "Cookbook", "Biography", "Journal", "Guidebook",
              "Poetry Collection", "Textbook"],
    "Toys & Games": ["Board Game", "Puzzle", "Action Figure", "Building Blocks",
                     "Remote Car", "Plush Toy"],
    "Grocery": ["Organic Snack Pack", "Coffee Beans", "Tea Set", "Spice Mix",
               "Granola Bars", "Olive Oil"],
}

DESC_TEMPLATES = [
    "A {adj} {noun} designed for everyday {cat} needs.",
    "Experience the {adj} {noun} - a customer favorite in {cat}.",
    "This {adj} {noun} combines quality and value for {cat} shoppers.",
    "Upgrade your routine with this {adj} {noun}.",
    "{adj} {noun} built to last, loved by thousands of buyers.",
]


def generate_products(n=N_PRODUCTS):
    rows = []
    for pid in range(1, n + 1):
        cat = RNG.choice(CATEGORY_LIST)
        brand = RNG.choice(CATEGORIES[cat])
        adj = RNG.choice(ADJECTIVES)
        noun = RNG.choice(NOUNS[cat])
        title = f"{adj} {noun}"
        template = RNG.choice(DESC_TEMPLATES)
        description = template.format(adj=adj.lower(), noun=noun.lower(), cat=cat.lower())
        # price depends loosely on category to feel realistic
        base_price = {
            "Electronics": 120, "Fashion": 45, "Home & Kitchen": 60,
            "Beauty & Personal Care": 25, "Sports & Outdoors": 55,
            "Books": 18, "Toys & Games": 30, "Grocery": 15,
        }[cat]
        price = round(max(3, RNG.gamma(shape=2.2, scale=base_price / 2.2)), 2)
        # some brands/descriptions randomly missing -> realistic messiness
        if RNG.random() < 0.03:
            brand = None
        if RNG.random() < 0.02:
            description = None
        tags = ",".join(RNG.choice([adj, cat.split()[0], noun, "bestseller", "new"],
                                    size=RNG.integers(1, 4), replace=False))
        rows.append({
            "product_id": pid,
            "title": title,
            "category": cat,
            "brand": brand,
            "price": price,
            "description": description,
            "tags": tags,
        })
    return pd.DataFrame(rows)


def generate_users(n=N_USERS):
    rows = []
    for uid in range(1, n + 1):
        # each user has 1-3 preferred categories (drives correlated behavior)
        n_pref = RNG.integers(1, 4)
        prefs = RNG.choice(CATEGORY_LIST, size=n_pref, replace=False)
        signup_offset = int(RNG.integers(30, 900))
        rows.append({
            "user_id": uid,
            "preferred_categories": ",".join(prefs),
            "signup_days_ago": signup_offset,
        })
    return pd.DataFrame(rows)


def generate_interactions(users_df, products_df, n=N_INTERACTIONS):
    # Zipf-like popularity: assign each product a popularity weight
    popularity = RNG.zipf(a=1.6, size=len(products_df)).astype(float)
    popularity = popularity / popularity.sum()
    prod_ids = products_df["product_id"].values
    prod_cat = dict(zip(products_df.product_id, products_df.category))

    interaction_types = ["view", "add_to_cart", "wishlist", "purchase"]
    # funnel probabilities: most interactions are views
    type_probs = [0.60, 0.20, 0.10, 0.10]

    rows = []
    user_ids = users_df["user_id"].values
    user_prefs = dict(zip(users_df.user_id, users_df.preferred_categories))

    start_ts = pd.Timestamp("2024-01-01")
    for _ in range(n):
        uid = int(RNG.choice(user_ids))
        prefs = user_prefs[uid].split(",")
        # 75% chance user interacts within a preferred category, else explores
        if RNG.random() < 0.75:
            cat = RNG.choice(prefs)
            cat_products = products_df.loc[products_df.category == cat, "product_id"].values
            if len(cat_products) == 0:
                pid = int(RNG.choice(prod_ids, p=popularity))
            else:
                weights = popularity[np.isin(prod_ids, cat_products)]
                weights = weights / weights.sum()
                pid = int(RNG.choice(cat_products, p=weights))
        else:
            pid = int(RNG.choice(prod_ids, p=popularity))

        itype = RNG.choice(interaction_types, p=type_probs)
        day_offset = int(RNG.integers(0, 540))
        ts = start_ts + pd.Timedelta(days=day_offset, hours=int(RNG.integers(0, 24)))

        rating = np.nan
        if itype == "purchase" and RNG.random() < 0.55:
            # rating skewed toward 4-5 stars, realistic bias
            rating = int(RNG.choice([1, 2, 3, 4, 5], p=[0.04, 0.06, 0.15, 0.35, 0.40]))

        rows.append({
            "user_id": uid,
            "product_id": pid,
            "interaction_type": itype,
            "rating": rating,
            "timestamp": ts,
        })

    df = pd.DataFrame(rows)
    # inject a few duplicate rows and a few invalid ratings to simulate messy real data
    dup_sample = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_sample], ignore_index=True)
    noise_idx = df.sample(frac=0.002, random_state=2).index
    df.loc[noise_idx, "rating"] = 9  # unrealistic rating to be cleaned later
    return df


def main():
    print("Generating products...")
    products = generate_products()
    print("Generating users...")
    users = generate_users()
    print("Generating interactions...")
    interactions = generate_interactions(users, products)

    products.to_csv(RAW_DIR / "products.csv", index=False)
    users.to_csv(RAW_DIR / "users.csv", index=False)
    interactions.to_csv(RAW_DIR / "interactions.csv", index=False)

    print(f"Saved {len(products)} products, {len(users)} users, "
          f"{len(interactions)} interactions to {RAW_DIR}")


if __name__ == "__main__":
    main()
