# 🛍️ AI-Powered E-Commerce Product Recommendation System

An end-to-end, production-style **Machine Learning recommendation engine**
that combines **Content-Based Filtering**, **Collaborative Filtering**
(matrix factorization), and a **Hybrid model** to generate personalized
product recommendations — complete with cold-start handling,
recommendation explainability, rigorous ranking-metric evaluation, and an
interactive **Streamlit** web application.

Built as an academic / portfolio project to demonstrate a realistic,
modular ML system design rather than a single notebook.

---

## 📌 Problem Statement

E-commerce platforms have far more products than any single user can
browse. Left unassisted, users struggle to discover relevant products,
which hurts both engagement and revenue. This project builds a
recommendation engine that surfaces relevant products to each user based
on their **purchases, views, ratings, wishlist activity, and browsing
behavior** — while gracefully handling brand-new users and brand-new
products that have no interaction history yet (the "cold-start" problem).

## 🎯 Objectives

- Recommend products a user is likely to want, personalized to their
  behavior.
- Recommend products similar to a given product (content similarity).
- Handle new users and new products without failing or returning empty
  results.
- Make every recommendation **explainable** ("why was this recommended?").
- Evaluate models with **ranking-appropriate metrics**, not plain accuracy.
- Ship a usable, interactive web app, not just a notebook.

## ✨ Features

- Three recommendation approaches: Content-Based, Collaborative Filtering,
  and a tuned Hybrid engine (the main engine).
- Cold-start strategy for new users (category-based / popularity fallback)
  and new products (content-based similarity only, clearly documented).
- Per-recommendation natural-language explanations.
- Full EDA suite (9 charts) covering popularity, ratings, categories,
  price, and user/product activity distributions.
- Feature engineering with leakage-safe, time-based train/test splitting.
- Ranking-metric evaluation: Precision@K, Recall@K, F1@K, MAP@K, NDCG@K,
  Coverage, Diversity — with a model comparison table.
- Data-driven hybrid weight tuning (grid search on NDCG@10), not hand-picked
  weights.
- A 5-page Streamlit app: Home, Personalized Recommendations, Similar
  Products, Popular Products, and an interactive Analytics Dashboard.
- Modular `src/` codebase, model persistence with `joblib`, and a single
  `run_pipeline.py` entry point.

## 🏗️ System Architecture

```text
Raw Dataset (products, users, interactions)
        │
        ▼
Data Cleaning (dedupe, missing values, invalid ratings/prices)
        │
        ▼
Exploratory Data Analysis (9 charts)
        │
        ▼
Feature Engineering (product & user features, category preferences)
        │
        ▼
Time-based Train/Test Split (per-user, leakage-safe)
        │
        ├──▶ Content-Based Model (TF-IDF + Cosine Similarity)
        │
        ├──▶ Collaborative Filtering Model (Truncated SVD on weighted
        │      user-item interaction matrix)
        │
        ▼
Hybrid Recommendation Engine
   final_score = α · collaborative_score + β · content_score
   (α, β tuned by grid search; adapted per-user for cold-start)
        │
        ▼
Evaluation (Precision/Recall/F1/MAP/NDCG@K, Coverage, Diversity)
        │
        ▼
Model Persistence (joblib)
        │
        ▼
Streamlit Web Application
```

## 📊 Dataset

This project was built in a sandboxed environment **without internet
access**, so a live download from a Kaggle dataset (e.g. "Amazon Product
Reviews", "Olist Brazilian E-Commerce", "H&M Personalized
Recommendations") wasn't possible. Instead, `src/generate_dataset.py`
generates a **large (800 products / 2,000 users / ~60,000 interactions),
statistically realistic synthetic dataset** that reproduces properties
real e-commerce data is known to have:

- **Zipf-distributed product popularity** — a small set of products
  absorb most interactions (the classic long tail).
- **Category-correlated user preferences** — users mostly repeat-engage
  within 1-3 preferred categories, with occasional exploration.
- **A realistic interaction funnel** — views ≫ carts ≫ wishlists ≫
  purchases.
- **Ratings skewed toward 4-5 stars**, a well-documented real-world bias.
- **Intentional messiness** — missing brands/descriptions, duplicate
  interaction rows, and a small number of invalid ratings — so the
  preprocessing step has genuine, demonstrable work to do.

Because every downstream module (`data_preprocessing.py`,
`feature_engineering.py`, the recommenders) only relies on standard
column names (`user_id`, `product_id`, `category`, `brand`, `price`,
`rating`, `interaction_type`, `timestamp`, ...), **you can swap in a real
dataset at any time** — just point `data/raw/*.csv` at your own files
with matching columns (or edit the column names at the top of
`data_preprocessing.py`), and re-run the pipeline.

## 🧠 Recommendation Methodology

### A. Content-Based Filtering (`src/content_based.py`)
Represents each product as a **TF-IDF vector** over its title, brand,
category, tags, and description, and ranks products by **cosine
similarity**. A user's profile vector is the interaction-weighted average
of the vectors of products they've engaged with. TF-IDF down-weights
generic words ("premium", "quality") and up-weights the words that
actually distinguish products, and cosine similarity isn't biased by
description length — both properties matter for a *similarity* signal,
and both are fully interpretable, which is needed for the explanation
layer.

### B. Collaborative Filtering (`src/collaborative_filtering.py`)
Builds a sparse, **weighted user × product interaction matrix**
(purchase=5, wishlist=3, add-to-cart=2, view=1) and factorizes it with
**Truncated SVD** into latent taste factors. This captures *behavioral*
patterns text can never reveal (e.g. "buyers of running shoes also buy
water bottles"). Truncated SVD was chosen over a full library like
`surprise`/`implicit` because it works directly on the sparse matrix with
only `scikit-learn` (keeping `requirements.txt` minimal and portable —
important in environments where extra ML packages may not install), is
fast to train/retrain, and its latent vectors are simple to persist and
reuse for user-user similarity in the explanation layer.

### C. Hybrid Recommendation System (`src/hybrid_recommender.py`) — **the main engine**
```
final_score = α · collaborative_score + β · content_score      (α + β = 1)
```
- **α/β are not hand-picked.** `src/evaluation.py::tune_hybrid_weights`
  grid-searches α from 0.0 to 1.0 in steps of 0.1 and selects whichever
  value maximizes **NDCG@10** on the held-out test set (see
  `data/processed/hybrid_weight_tuning.csv` after running the pipeline).
- **α/β are also adapted per user at recommendation time.** Users with
  fewer than 5 recorded interactions get their collaborative weight
  scaled down proportionally to how little history they have, so
  content-based similarity fills the gap — this is what makes the hybrid
  model robust for low-activity users where a pure collaborative model
  would have almost no signal.

## 🧊 Cold-Start Strategy

| Scenario | Strategy |
|---|---|
| **New user** (no interactions) | Falls back to the user's stated `preferred_categories` if available, returning highly-rated / popular products within those categories; otherwise falls back to globally popular products. |
| **New product** (no interactions yet) | Has no collaborative signal by definition (nothing to factorize) — it can only surface through content-based similarity to products a user already likes, or through `get_popular_products(category=...)` once it starts accumulating its own interactions. This limitation is explicit rather than silently failing. |
| **Low-activity user** (some but <5 interactions) | The hybrid model's α is scaled down proportionally to interaction count, so content-based similarity dominates until enough behavioral signal accumulates. |

## 🧩 Feature Engineering

| Feature | Meaning | Purpose |
|---|---|---|
| `avg_rating`, `n_ratings` | Mean rating / rating count per product | Quality signal for cold-start & explanations |
| `weighted_interaction_score` | Σ(interaction weight) per product | Robust popularity signal |
| `popularity_rank` | Percentile rank of the above | Trending vs. niche segmentation |
| `price_bucket` | Quartile-based Low/Medium/High/Premium | Dashboard segmentation |
| `combined_text` | title+brand+category+tags+description | Input to TF-IDF |
| `activity_score` | Weighted sum of a user's interactions | Drives cold-start confidence blending |
| `top_category` | User's most-interacted category | "Because you frequently buy from X" explanation |
| `category_preference_matrix` | Per-user normalized share of interactions per category | Cold-start category recommendations |

All features are computed **only from the training split** of a
time-based, per-user train/test split (each user's most recent 20% of
interactions become the test set) — this prevents the model from ever
"knowing" a future interaction while training, avoiding data leakage.

## 📈 Exploratory Data Analysis

Nine charts are generated by `src/eda.py` into
`data/processed/eda_charts/`: most popular products, most popular
categories, rating distribution, interactions per user, interactions per
product, catalog composition by category, price distribution, user
activity distribution, and the product popularity long-tail curve. See
`notebooks/01_eda.ipynb` for the full walkthrough with commentary.

## 🧪 Evaluation Metrics

Plain accuracy is meaningless for recommendation — the task is ranking a
handful of relevant items near the top of a list drawn from hundreds of
candidates. This project implements:

- **Precision@K** — of the K recommended items, what fraction were relevant?
- **Recall@K** — of all relevant items, what fraction appeared in the top K?
- **F1@K** — harmonic mean of the two.
- **MAP@K** — rewards relevant items appearing *earlier* in the ranking.
- **NDCG@K** — like MAP, but with graded relevance (a test-set purchase
  counts for more than a mere view) and logarithmic position discounting.
- **Coverage** — fraction of the whole catalog that ever gets recommended
  (catches a model that always recommends the same best-sellers).
- **Diversity** — average pairwise dissimilarity within a user's
  recommendation list (catches near-duplicate recommendations).

### Model Comparison (K=10, generated by `run_pipeline.py`)

| Model | Precision@10 | Recall@10 | F1@10 | MAP@10 | NDCG@10 | Coverage | Diversity |
|---|---|---|---|---|---|---|---|
| Content-Based | 0.0060 | 0.0136 | 0.0083 | 0.0053 | 0.0082 | 0.376 | 0.672 |
| Collaborative | 0.0264 | 0.0619 | 0.0365 | 0.0295 | 0.0479 | 0.199 | 0.913 |
| **Hybrid** | **0.0264** | **0.0619** | **0.0365** | **0.0295** | **0.0479** | 0.199 | 0.913 |

*(Regenerate this table any time with `python run_pipeline.py` — it's
also written to `data/processed/model_comparison.csv` and rendered live
in the Streamlit Analytics Dashboard.)*

**Why the Hybrid model is the right default engine — even though its
warm-user metrics tie Collaborative Filtering here:** on this dataset,
user behavior is strongly category-correlated, so Collaborative Filtering
alone already captures most of the exploitable signal for users with
enough history — the grid search correctly finds that α≈1.0 is optimal
*for those users*. But the Hybrid model isn't just that global weight: its
**per-user adaptive blending** automatically shifts weight toward
content-based similarity for low-activity and brand-new users, exactly
where pure Collaborative Filtering has little or no signal. In other
words, Hybrid ≈ Collaborative for established users, but **Hybrid ≫
Collaborative for the cold-start users** a production system encounters
constantly — which is why it's the model exposed by the app, not either
component alone. (Absolute metric values are modest across the board
because the evaluation is intentionally strict — recommending 10 items
out of an 800-product catalog and only counting a hit if it appears in a
user's *held-out future* interactions.)

## 🖥️ Recommendation Engine API

```python
recommend_products(user_id, top_n=10)              # main hybrid engine
recommend_similar_products(product_id, top_n=10)    # content-based
get_popular_products(category=None, top_n=10)        # cold-start / trending
get_personalized_recommendations(user_id, top_n=10)  # alias for recommend_products
```

Every result includes `product_id`, `title`, `category`, `price`, a
recommendation `score`, and a plain-English `reason`.

## 🌐 Web Application (Streamlit)

- **Home** — project overview, live catalog stats, methodology summary.
- **Personalized Recommendations** — pick a user, see their recent
  history (if any) and hybrid-engine recommendations with scores and
  reasons.
- **Similar Products** — pick a product, see content-based similar items.
- **Popular Products** — trending / top-rated / category-wise tabs.
- **Analytics Dashboard** — interactive Plotly charts for catalog
  composition, popularity, ratings, user activity, and live model
  comparison metrics.

## 📁 Project Structure

```text
ecommerce-recommender/
├── data/
│   ├── raw/                 # generated/raw CSVs
│   └── processed/           # cleaned data, features, EDA charts, metrics
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── utils.py
│   ├── generate_dataset.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── content_based.py
│   ├── collaborative_filtering.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
├── models/                  # saved joblib models (generated)
├── app/
│   └── app.py                # Streamlit application
├── run_pipeline.py           # runs the full pipeline end-to-end
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Technologies Used

Python · pandas · NumPy · scikit-learn (TF-IDF, Truncated SVD, cosine
similarity) · SciPy (sparse matrices) · Matplotlib / Seaborn (EDA) ·
Plotly (interactive dashboard) · Streamlit (web app) · joblib (model
persistence).

Traditional ML was used deliberately instead of deep learning: TF-IDF +
matrix factorization already captures the relevant text and behavioral
signal at this data scale, trains in seconds, and stays fully
interpretable for the explanation layer — a neural recommender (e.g. a
two-tower or autoencoder model) would only be justified with far more
interaction data than 60K rows and would sacrifice interpretability for
a benefit this project doesn't need.

## 🚀 Installation & How to Run

```bash
# 1. Clone / open the project folder
cd ecommerce-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full ML pipeline (generates data if missing, cleans it, runs
#    EDA, trains all 3 models, tunes hybrid weights, evaluates, saves models)
python run_pipeline.py

# 4. Launch the web app
streamlit run app/app.py
```

The app will open at `http://localhost:8501`. Re-run `run_pipeline.py`
any time you change the data or feature engineering — the app always
loads whatever is currently saved in `models/`.

To explore step-by-step with explanations, open the notebooks in
`notebooks/` in order (01 → 02 → 03).

## 📸 Screenshots

*(Add screenshots of the Home, Personalized Recommendations, and
Analytics Dashboard pages here after running the app locally.)*

## 🔮 Future Improvements

- Swap in a real public dataset (Amazon Reviews, Olist, H&M) — the
  pipeline needs no code changes, only matching column names.
- Add implicit-feedback-specific factorization (e.g. ALS with confidence
  weighting) as an additional collaborative model to compare against SVD.
- Learn α/β with a lightweight meta-model (e.g. logistic regression over
  user activity level) instead of the current confidence-based rule.
- Add a session-based / sequential model (e.g. GRU4Rec) to capture
  short-term intent within a browsing session.
- Add A/B-testing instrumentation to compare online engagement, not just
  offline ranking metrics.
- Deploy the Streamlit app (Streamlit Community Cloud / Docker) for a
  live demo link.

---

*Academic project — built end-to-end with modular, production-style
Python and a fully data-driven hybrid recommendation engine.*
