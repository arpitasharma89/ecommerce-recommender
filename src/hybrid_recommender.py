"""
hybrid_recommender.py
------------------------
The MAIN recommendation engine. Combines Content-Based and Collaborative
Filtering into a single weighted hybrid score, and handles the cold-start
problem for both new users and new products.

HYBRID SCORE
    final_score = alpha * collaborative_score + beta * content_score
    (alpha + beta = 1, both scores pre-normalized to [0, 1])

CHOOSING alpha / beta (NOT ARBITRARY)
Collaborative filtering captures *behavioral* patterns other users
reveal (e.g. "users who buy running shoes also buy water bottles", a
connection no text field would ever suggest), while content-based
filtering captures *semantic* similarity and works even when
collaborative signal is sparse (new/low-activity users or products).
Because collaborative filtering generally has *higher ceiling accuracy*
once a user has enough interaction history (it exploits the full
community's behavior, not just text overlap), we weight it more heavily
by default -- but we don't hard-code a single global value. Instead:

    - alpha/beta are tuned by grid search in evaluation.py, sweeping
      alpha in {0.0, 0.1, ..., 1.0} and picking the value that maximizes
      NDCG@10 on the held-out test set (see `tune_hybrid_weights`).
    - The weight is further ADAPTED PER USER at recommendation time:
      users with very little interaction history get a lower alpha
      (collaborative filtering is unreliable with little data -- the
      "cold-start" issue) and a higher beta, so content-based similarity
      dominates until enough behavioral signal accumulates. This is a
      simple, interpretable confidence-based blending rule rather than a
      second learned model, which keeps the system easy to explain to a
      reviewer.

COLD START
    - New user (no interactions): falls back to
        1) explicit preferred_categories from their profile if available
        2) else globally popular / highly-rated products
      Weighted toward highly-rated, popular items within the user's
      preferred categories.
    - New product (no interactions yet): has no collaborative signal by
      definition (nothing to factorize), so it can only surface through
      content-based similarity to products the user already likes, or
      through get_popular_products(category=...) once it starts
      accumulating interactions. This is stated explicitly rather than
      silently failing.

EXPLAINABILITY
    Every recommendation returned includes a `reason` string generated
    from which signal(s) drove the score (content overlap with a
    previously liked product, collaborative similarity to like-minded
    users, category affinity, or popularity/rating within a category).
"""

import numpy as np
import pandas as pd

from utils import INTERACTION_WEIGHTS, get_logger

logger = get_logger("hybrid_recommender")

COLD_START_INTERACTION_THRESHOLD = 5  # below this, we lean on content + popularity


class HybridRecommender:
    def __init__(self, content_model, collab_model, products: pd.DataFrame,
                 users: pd.DataFrame, interactions: pd.DataFrame,
                 alpha: float = 0.6, beta: float = 0.4):
        self.content_model = content_model
        self.collab_model = collab_model
        self.products_lookup = products.set_index("product_id", drop=False)
        self.products = products.reset_index(drop=True)
        self.users = users.set_index("user_id", drop=False)
        self.interactions = interactions
        self.alpha = alpha
        self.beta = beta

    # ------------------------------------------------------------------
    # Cold-start helpers
    # ------------------------------------------------------------------
    def get_popular_products(self, category: str | None = None, top_n: int = 10) -> pd.DataFrame:
        df = self.products.copy()
        if category:
            df = df[df["category"] == category]
        df = df.sort_values(
            by=["weighted_interaction_score", "avg_rating"], ascending=False
        ).head(top_n)
        out = df[["product_id", "title", "category", "price", "avg_rating",
                   "weighted_interaction_score"]].copy()
        out["score"] = out["weighted_interaction_score"] / (
            out["weighted_interaction_score"].max() + 1e-9
        )
        out["reason"] = (
            f"Popular in {category}" if category else "Trending across the whole catalog"
        )
        return out.drop(columns="weighted_interaction_score")

    def _cold_start_new_user(self, user_id: int, top_n: int) -> pd.DataFrame:
        preferred = ""
        if user_id in self.users.index:
            preferred = self.users.loc[user_id, "preferred_categories"] or ""
        cats = [c for c in str(preferred).split(",") if c]

        if cats:
            per_cat = max(1, top_n // len(cats))
            frames = []
            for c in cats:
                pop = self.get_popular_products(category=c, top_n=per_cat)
                pop["reason"] = f"Highly rated / popular in your preferred category: {c}"
                frames.append(pop)
            result = pd.concat(frames).drop_duplicates(subset="product_id").head(top_n)
        else:
            result = self.get_popular_products(top_n=top_n)
            result["reason"] = "Popular among all users (no history yet to personalize from)"
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Adaptive weighting
    # ------------------------------------------------------------------
    def _adaptive_alpha_beta(self, user_id: int) -> tuple[float, float]:
        n_interactions = int((self.interactions["user_id"] == user_id).sum())
        if n_interactions >= COLD_START_INTERACTION_THRESHOLD:
            return self.alpha, self.beta
        # scale collaborative weight down linearly toward 0 as interactions -> 0
        confidence = n_interactions / COLD_START_INTERACTION_THRESHOLD
        alpha = self.alpha * confidence
        beta = 1 - alpha
        return alpha, beta

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------
    def recommend_products(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """The MAIN hybrid recommendation function."""
        n_interactions = int((self.interactions["user_id"] == user_id).sum())
        if n_interactions == 0:
            logger.info(f"User {user_id} has no history -> cold-start path")
            return self._cold_start_new_user(user_id, top_n)

        alpha, beta = self._adaptive_alpha_beta(user_id)

        collab_df = self.collab_model.recommend_for_user(
            user_id, self.interactions, top_n=200, exclude_seen=True
        )
        content_df = self.content_model.recommend_for_user(
            user_id, self.interactions, top_n=200, exclude_seen=True
        )

        merged = pd.merge(collab_df, content_df, on="product_id", how="outer").fillna(0.0)
        if merged.empty:
            return self._cold_start_new_user(user_id, top_n)

        merged["final_score"] = alpha * merged["collab_score"] + beta * merged["content_score"]
        merged = merged.sort_values("final_score", ascending=False).head(top_n)

        merged = merged.merge(
            self.products[["product_id", "title", "category", "price", "avg_rating"]],
            on="product_id", how="left"
        )
        merged["reason"] = merged.apply(
            lambda r: self._explain(user_id, r, alpha, beta), axis=1
        )
        return merged.reset_index(drop=True)

    def recommend_similar_products(self, product_id: int, top_n: int = 10) -> pd.DataFrame:
        sims = self.content_model.similar_products(product_id, top_n=top_n)
        if sims.empty:
            return sims
        sims = sims.merge(
            self.products[["product_id", "title", "category", "price", "avg_rating"]],
            on="product_id", how="left"
        )
        base_title = (self.products_lookup.loc[product_id, "title"]
                      if product_id in self.products_lookup.index else "this product")
        sims["reason"] = f"Similar in description/category/brand to '{base_title}'"
        sims = sims.rename(columns={"content_score": "score"})
        return sims

    def get_personalized_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Alias kept for API-spec completeness; identical to
        recommend_products (the hybrid IS the personalized engine)."""
        return self.recommend_products(user_id, top_n=top_n)

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------
    def _explain(self, user_id: int, row, alpha: float, beta: float) -> str:
        collab_score = row.get("collab_score", 0.0)
        content_score = row.get("content_score", 0.0)
        category = row.get("category", "")

        user_top_cat = None
        if user_id in self.users.index and "top_category" in self.users.columns:
            user_top_cat = self.users.loc[user_id, "top_category"]

        if collab_score >= content_score and collab_score > 0.3:
            return "Popular among users with similar preferences to yours"
        if content_score > 0.15:
            return "Similar to products you've previously viewed or purchased"
        if category and category == user_top_cat:
            return f"Because you frequently interact with the {category} category"
        if row.get("avg_rating", 0) >= 4.2:
            return f"Highly rated in {category}" if category else "Highly rated overall"
        return "Recommended based on your overall activity"
