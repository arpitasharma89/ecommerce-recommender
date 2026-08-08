"""
collaborative_filtering.py
----------------------------
Collaborative Filtering using Matrix Factorization (Truncated SVD) over an
implicit, weighted user-item interaction matrix.

WHAT IT DOES
Builds a sparse user x product matrix where each cell is the weighted sum
of that user's interactions with that product (purchase=5, wishlist=3,
add_to_cart=2, view=1 -- see utils.INTERACTION_WEIGHTS). Matrix
factorization decomposes this matrix into a small number of *latent
factors* that capture hidden patterns of taste (e.g. "tends to like
budget electronics" or "prefers premium home goods") shared across users,
which is what lets the model recommend a product a user has never
interacted with, based on what similar users liked.

WHY TRUNCATED SVD (over, say, a full Surprise/implicit-ALS library)
- The project's interaction data is *implicit and weighted* (views, carts,
  wishlists, purchases) rather than explicit 1-5 star ratings for every
  pair, which is exactly the setting matrix factorization on a weighted
  interaction matrix was designed for.
- scikit-learn's TruncatedSVD works directly and efficiently on the large,
  sparse user-item matrix without needing extra dependencies (Surprise /
  implicit are not always available in restricted environments), keeping
  the project's requirements.txt minimal and portable.
- SVD-based factorization is fast to train/retrain, easy to persist with
  joblib, and the resulting latent vectors are simple to reuse for
  user-user or item-item similarity, which the hybrid model needs anyway.

HOW IT WORKS HERE
1. Pivot train interactions into a sparse (n_users x n_products) weighted
   matrix.
2. Fit TruncatedSVD to get U (user latent factors) and Vt (product latent
   factors).
3. Reconstruct predicted affinity scores as U @ Vt, i.e. a dense matrix of
   "how much would this user like this product" scores.
4. For a given user, rank all products by predicted score, excluding ones
   already interacted with.

WHAT THE OUTPUT MEANS
The predicted score is a relative affinity score (not a 1-5 rating) --
higher means the model believes the user-product pair is a stronger
match, based on patterns learned from all other users' behavior. Scores
are min-max normalized to [0, 1] before being combined in the hybrid
model so they're on a comparable scale to content similarity.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from utils import INTERACTION_WEIGHTS, RANDOM_SEED, get_logger

logger = get_logger("collaborative_filtering")


class CollaborativeRecommender:
    def __init__(self, n_factors: int = 30):
        self.n_factors = n_factors
        self.svd = None
        self.user_ids = None
        self.product_ids = None
        self.user_id_to_idx = None
        self.product_id_to_idx = None
        self.user_factors = None
        self.product_factors = None

    def fit(self, interactions: pd.DataFrame):
        interactions = interactions.copy()
        interactions["weight"] = interactions["interaction_type"].map(INTERACTION_WEIGHTS)
        matrix_df = interactions.groupby(["user_id", "product_id"])["weight"].sum().reset_index()

        self.user_ids = sorted(matrix_df["user_id"].unique())
        self.product_ids = sorted(matrix_df["product_id"].unique())
        self.user_id_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.product_id_to_idx = {p: i for i, p in enumerate(self.product_ids)}

        rows = matrix_df["user_id"].map(self.user_id_to_idx)
        cols = matrix_df["product_id"].map(self.product_id_to_idx)
        sparse_matrix = csr_matrix(
            (matrix_df["weight"], (rows, cols)),
            shape=(len(self.user_ids), len(self.product_ids)),
        )

        n_components = min(self.n_factors, min(sparse_matrix.shape) - 1)
        n_components = max(2, n_components)
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        self.user_factors = self.svd.fit_transform(sparse_matrix)
        self.product_factors = self.svd.components_.T

        explained = self.svd.explained_variance_ratio_.sum()
        logger.info(f"Fitted TruncatedSVD with {n_components} latent factors "
                    f"(explains {explained:.1%} of interaction variance) on "
                    f"{len(self.user_ids)} users x {len(self.product_ids)} products")
        return self

    def predict_scores_for_user(self, user_id: int) -> np.ndarray | None:
        if user_id not in self.user_id_to_idx:
            return None
        u_idx = self.user_id_to_idx[user_id]
        scores = self.user_factors[u_idx] @ self.product_factors.T
        return scores

    def recommend_for_user(self, user_id: int, interactions: pd.DataFrame,
                            top_n: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        scores = self.predict_scores_for_user(user_id)
        if scores is None:
            return pd.DataFrame(columns=["product_id", "collab_score"])

        # min-max normalize to [0, 1] for comparability with content scores
        s_min, s_max = scores.min(), scores.max()
        norm_scores = (scores - s_min) / (s_max - s_min + 1e-9)

        seen = set(interactions.loc[interactions["user_id"] == user_id, "product_id"]) if exclude_seen else set()
        order = np.argsort(-norm_scores)
        results = []
        for idx in order:
            pid = self.product_ids[idx]
            if pid in seen:
                continue
            results.append((pid, float(norm_scores[idx])))
            if len(results) >= top_n:
                break
        return pd.DataFrame(results, columns=["product_id", "collab_score"])

    def similar_users(self, user_id: int, top_n: int = 10):
        """Used for the 'popular among users with similar preferences'
        explanation."""
        if user_id not in self.user_id_to_idx:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        u_idx = self.user_id_to_idx[user_id]
        sims = cosine_similarity(self.user_factors[u_idx].reshape(1, -1), self.user_factors).flatten()
        order = np.argsort(-sims)
        order = [i for i in order if i != u_idx][:top_n]
        return [self.user_ids[i] for i in order]
