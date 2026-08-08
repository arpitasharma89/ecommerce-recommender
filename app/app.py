"""
app.py
------
Streamlit web application for the E-Commerce Product Recommendation System.

Run (from the project root, AFTER running `python run_pipeline.py` once):
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from utils import PROCESSED_DIR, MODELS_DIR  # noqa: E402
from hybrid_recommender import HybridRecommender  # noqa: E402

st.set_page_config(
    page_title="E-Commerce Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .product-card {
        background: #ffffff; border-radius: 14px; padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 14px;
        border: 1px solid #eef0f3;
    }
    .score-badge {
        display: inline-block; background: #eef2ff; color: #4338ca;
        padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    }
    .reason-text { color: #6b7280; font-size: 0.85rem; font-style: italic; }
    .price-tag { color: #059669; font-weight: 700; font-size: 1.05rem; }
    .metric-card {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
        border-radius: 14px; padding: 18px; color: white; text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    content_model = joblib.load(MODELS_DIR / "content_model.joblib")
    collab_model = joblib.load(MODELS_DIR / "collab_model.joblib")
    weights = joblib.load(MODELS_DIR / "hybrid_weights.joblib")
    return content_model, collab_model, weights


@st.cache_data(show_spinner="Loading data...")
def load_data():
    products = pd.read_csv(MODELS_DIR / "products_features.csv")
    users = pd.read_csv(PROCESSED_DIR / "users_features.csv")
    train = pd.read_csv(MODELS_DIR / "interactions_train.csv", parse_dates=["timestamp"])
    return products, users, train


@st.cache_data(show_spinner=False)
def load_comparison():
    path = PROCESSED_DIR / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


def models_ready():
    required = [
        MODELS_DIR / "content_model.joblib",
        MODELS_DIR / "collab_model.joblib",
        MODELS_DIR / "hybrid_weights.joblib",
        MODELS_DIR / "products_features.csv",
        MODELS_DIR / "interactions_train.csv",
    ]
    return all(p.exists() for p in required)


if not models_ready():
    st.error(
        "Trained models were not found. Please run `python run_pipeline.py` "
        "from the project root first, then relaunch the app."
    )
    st.stop()

content_model, collab_model, weights = load_models()
products, users, train = load_data()
hybrid_model = HybridRecommender(content_model, collab_model, products, users, train, **weights)
comparison_df = load_comparison()

VALID_USER_IDS = sorted(users["user_id"].unique().tolist())
VALID_PRODUCT_IDS = sorted(products["product_id"].unique().tolist())


def render_products(df: pd.DataFrame, score_col: str = None, cols_per_row: int = 5):
    if df is None or df.empty:
        st.info("No recommendations available.")
        return
    rows = [df.iloc[i:i + cols_per_row] for i in range(0, len(df), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (_, item) in zip(cols, row.iterrows()):
            with col:
                score_html = ""
                if score_col and score_col in item and pd.notna(item[score_col]):
                    score_html = f'<span class="score-badge">Score: {item[score_col]:.2f}</span>'
                reason = item.get("reason", "")
                st.markdown(f"""
                <div class="product-card">
                    <b>{item.get('title', 'Unknown')}</b><br/>
                    <span style="color:#6b7280;font-size:0.85rem">{item.get('category','')}</span><br/>
                    <span class="price-tag">${item.get('price', 0):.2f}</span>
                    &nbsp;&nbsp;{score_html}<br/>
                    <span class="reason-text">💡 {reason}</span>
                </div>
                """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🛍️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Personalized Recommendations", "🔗 Similar Products",
     "🔥 Popular Products", "📊 Analytics Dashboard"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Catalog:** {len(products)} products")
st.sidebar.markdown(f"**Users:** {len(users)}")
st.sidebar.markdown(f"**Interactions:** {len(train)}")
st.sidebar.markdown(f"**Hybrid weights:** α={weights['alpha']:.2f} (collaborative), "
                     f"β={weights['beta']:.2f} (content)")

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🛍️ AI-Powered E-Commerce Product Recommendation System")
    st.markdown(
        "An intelligent recommendation engine that blends **content-based "
        "filtering**, **collaborative filtering (matrix factorization)**, "
        "and a **hybrid model** to recommend products to users based on "
        "their purchases, views, ratings, and browsing behavior."
    )

    c1, c2, c3, c4 = st.columns(4)
    for c, label, value in zip(
        [c1, c2, c3, c4],
        ["Products", "Users", "Interactions", "Categories"],
        [len(products), len(users), len(train), products['category'].nunique()],
    ):
        with c:
            st.markdown(f"""<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>""",
                        unsafe_allow_html=True)

    st.markdown("### How the Recommendation Engine Works")
    st.markdown("""
1. **Content-Based Filtering** — TF-IDF + cosine similarity over product
   title, brand, category, tags and description to find textually similar
   products.
2. **Collaborative Filtering** — Matrix factorization (Truncated SVD) over
   a weighted user-product interaction matrix (purchase > wishlist >
   add-to-cart > view) to learn latent taste patterns shared across users.
3. **Hybrid Engine** — `final_score = α · collaborative_score + β · content_score`,
   with weights tuned by grid search to maximize NDCG@10, and adapted
   per-user so low-activity users lean more on content-based similarity
   (cold-start handling).
    """)

    st.markdown("### Explore")
    st.markdown(
        "Use the sidebar to get **personalized recommendations** for a "
        "user, find **similar products**, browse what's **popular**, or "
        "explore the **analytics dashboard**."
    )

# ---------------------------------------------------------------------------
# PERSONALIZED RECOMMENDATIONS
# ---------------------------------------------------------------------------
elif page == "🎯 Personalized Recommendations":
    st.title("🎯 Personalized Recommendations")
    st.caption("Powered by the Hybrid Recommendation Engine")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.selectbox(
            "Select a User ID", VALID_USER_IDS,
            index=0,
            help="Pick a user to see recommendations tailored to their history."
        )
    with col2:
        top_n = st.slider("Number of recommendations", 5, 20, 10)

    user_hist = train[train["user_id"] == user_id]
    n_hist = len(user_hist)

    if n_hist == 0:
        st.warning(f"User {user_id} has no interaction history — showing **cold-start** "
                   f"recommendations (preferred categories / popular products).")
    else:
        st.success(f"User {user_id} has {n_hist} recorded interactions. "
                   f"Recommendations are personalized using the hybrid engine.")
        with st.expander("View this user's recent interactions"):
            recent = user_hist.merge(products[["product_id", "title", "category"]],
                                      on="product_id", how="left").sort_values("timestamp", ascending=False)
            st.dataframe(recent[["timestamp", "title", "category", "interaction_type", "rating"]].head(15),
                        use_container_width=True)

    with st.spinner("Generating recommendations..."):
        recs = hybrid_model.recommend_products(user_id, top_n=top_n)

    st.markdown("### Recommended for You")
    score_col = "final_score" if "final_score" in recs.columns else None
    render_products(recs, score_col=score_col)

# ---------------------------------------------------------------------------
# SIMILAR PRODUCTS
# ---------------------------------------------------------------------------
elif page == "🔗 Similar Products":
    st.title("🔗 Similar Products")
    st.caption("Powered by Content-Based Filtering (TF-IDF + Cosine Similarity)")

    product_titles = products.set_index("product_id")["title"]
    selected_title = st.selectbox(
        "Select a product",
        options=VALID_PRODUCT_IDS,
        format_func=lambda pid: f"{product_titles.get(pid, pid)} (#{pid})",
    )
    top_n = st.slider("Number of similar products", 5, 20, 10, key="sim_slider")

    base = products[products["product_id"] == selected_title].iloc[0]
    st.markdown(f"**Base product:** {base['title']} — {base['category']} — ${base['price']:.2f}")

    with st.spinner("Finding similar products..."):
        sims = hybrid_model.recommend_similar_products(selected_title, top_n=top_n)

    st.markdown("### Similar Products")
    render_products(sims, score_col="score")

# ---------------------------------------------------------------------------
# POPULAR PRODUCTS
# ---------------------------------------------------------------------------
elif page == "🔥 Popular Products":
    st.title("🔥 Popular Products")

    tab1, tab2, tab3 = st.tabs(["Trending (All)", "Top Rated", "Category-wise"])

    with tab1:
        top_n = st.slider("Show top N", 5, 30, 10, key="pop_all")
        pop = hybrid_model.get_popular_products(top_n=top_n)
        render_products(pop, score_col="score")

    with tab2:
        top_n2 = st.slider("Show top N", 5, 30, 10, key="pop_rated")
        rated = products[products["n_ratings"] >= 3].sort_values(
            "avg_rating", ascending=False
        ).head(top_n2).copy()
        rated["reason"] = "Highly rated by customers"
        render_products(rated, score_col="avg_rating")

    with tab3:
        cat = st.selectbox("Category", sorted(products["category"].unique()))
        top_n3 = st.slider("Show top N", 5, 30, 10, key="pop_cat")
        cat_pop = hybrid_model.get_popular_products(category=cat, top_n=top_n3)
        render_products(cat_pop, score_col="score")

# ---------------------------------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------------------------------
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Catalog & Popularity", "Ratings & Activity", "User Behavior", "Model Performance"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            cat_counts = products["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.pie(cat_counts, names="category", values="count",
                        title="Catalog Composition by Category", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top_pop = products.sort_values("weighted_interaction_score", ascending=False).head(15)
            fig = px.bar(top_pop, x="weighted_interaction_score", y="title", orientation="h",
                        title="Top 15 Most Popular Products", color="category")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(products, x="price", nbins=40, title="Price Distribution",
                           color="category")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            rated_int = train.dropna(subset=["rating"])
            fig = px.histogram(rated_int, x="rating", title="Rating Distribution", nbins=5)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cat_rating = products[products["n_ratings"] > 0].groupby("category")["avg_rating"].mean().reset_index()
            fig = px.bar(cat_rating, x="category", y="avg_rating", title="Average Rating by Category")
            st.plotly_chart(fig, use_container_width=True)

        per_product = train.groupby("product_id").size().reset_index(name="n_interactions")
        fig = px.histogram(per_product, x="n_interactions", nbins=40,
                           title="Interactions per Product (Long Tail)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            per_user = train.groupby("user_id").size().reset_index(name="n_interactions")
            fig = px.histogram(per_user, x="n_interactions", nbins=40,
                               title="Interactions per User")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(users, x="activity_score", nbins=40,
                               title="User Activity Score Distribution")
            st.plotly_chart(fig, use_container_width=True)

        type_counts = train["interaction_type"].value_counts().reset_index()
        type_counts.columns = ["interaction_type", "count"]
        fig = px.bar(type_counts, x="interaction_type", y="count",
                    title="Interaction Type Funnel (View → Cart → Wishlist → Purchase)")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if comparison_df is not None:
            st.markdown("### Model Comparison (Precision/Recall/NDCG @ K=10)")
            st.dataframe(comparison_df.set_index("Model").style.format("{:.4f}", subset=[
                c for c in comparison_df.columns if c not in ("Model", "n_users_evaluated")
            ]), use_container_width=True)

            metrics_to_plot = ["Precision@K", "Recall@K", "F1@K", "MAP@K", "NDCG@K"]
            melted = comparison_df.melt(id_vars="Model", value_vars=metrics_to_plot,
                                        var_name="Metric", value_name="Value")
            fig = px.bar(melted, x="Metric", y="Value", color="Model", barmode="group",
                        title="Model Comparison Across Ranking Metrics")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(comparison_df, x="Model", y=["Coverage", "Diversity"], barmode="group",
                         title="Catalog Coverage & Recommendation Diversity")
            st.plotly_chart(fig2, use_container_width=True)

            best_model = comparison_df.loc[comparison_df["NDCG@K"].idxmax(), "Model"]
            st.success(f"**{best_model}** achieves the highest NDCG@10 on the held-out test set.")
        else:
            st.info("Run `python run_pipeline.py` to generate the model comparison table.")

st.sidebar.markdown("---")
st.sidebar.caption("Academic Project · Hybrid Recommendation Engine · Built with Python & Streamlit")
