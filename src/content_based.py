"""
content_based.py
-----------------
Content-Based Filtering using TF-IDF + Cosine Similarity.

WHAT IT DOES
Represents every product as a vector built from its text metadata
(title, brand, category, tags, description), weighted by TF-IDF
(Term Frequency - Inverse Document Frequency). Products with similar
vectors -> similar cosine similarity score -> considered "similar
products". A user's profile vector is built as the (recency- and
interaction-weighted) average of the vectors of products they've
interacted with, so we can recommend the products closest to that
profile.

WHY TF-IDF + COSINE SIMILARITY
- TF-IDF down-weights common words that appear in almost every product
  description (e.g. "quality", "premium") and up-weights the words that
  actually distinguish one product from another, which is exactly what
  we need for a *similarity* signal.
- Cosine similarity measures the angle between vectors rather than raw
  magnitude, so it isn't biased toward products with longer descriptions.
- Both are cheap to compute and fully interpretable, which matters for
  the "reason for recommendation" explanations this project needs to
  generate for every recommended product.

HOW IT WORKS HERE
1. Fit a TfidfVectorizer on `combined_text` for every product.
2. Compute pairwise cosine similarity between products (used directly
   for `recommend_similar_products`).
3. For `recommend_products`, build each user's profile vector as a
   weighted average of the TF-IDF vectors of products they interacted
   with (weight = INTERACTION_WEIGHTS[type]), then rank all products by
   cosine similarity to that profile.

WHAT THE OUTPUT MEANS
A score of 1.0 means "identical text profile"; 0.0 means "no shared
vocabulary at all". In practice, scores of 0.15-0.45 already indicate
meaningfully related products (same category/brand/keywords).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import INTERACTION_WEIGHTS, get_logger

logger = get_logger("content_based")


class ContentBasedRecommender:
    def __init__(self, products: pd.DataFrame):
        self.products = products.reset_index(drop=True)
        self.product_id_to_idx = {pid: i for i, pid in enumerate(self.products["product_id"])}
        self.idx_to_product_id = {i: pid for pid, i in self.product_id_to_idx.items()}
        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.products["combined_text"])
        logger.info(f"Fitted TF-IDF on {self.tfidf_matrix.shape[0]} products, "
                    f"{self.tfidf_matrix.shape[1]} vocabulary terms")

    def similar_products(self, product_id: int, top_n: int = 10) -> pd.DataFrame:
        if product_id not in self.product_id_to_idx:
            return pd.DataFrame(columns=["product_id", "score"])
        idx = self.product_id_to_idx[product_id]
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        order = np.argsort(-sims)
        order = [i for i in order if i != idx][:top_n]
        return pd.DataFrame({
            "product_id": [self.idx_to_product_id[i] for i in order],
            "content_score": [float(sims[i]) for i in order],
        })

    def user_profile_vector(self, user_id: int, interactions: pd.DataFrame):
        user_hist = interactions[interactions["user_id"] == user_id]
        if user_hist.empty:
            return None
        weights, vectors = [], []
        for _, row in user_hist.iterrows():
            pid = row["product_id"]
            if pid not in self.product_id_to_idx:
                continue
            w = INTERACTION_WEIGHTS.get(row["interaction_type"], 1.0)
            weights.append(w)
            vectors.append(self.tfidf_matrix[self.product_id_to_idx[pid]])
        if not vectors:
            return None
        weights = np.array(weights).reshape(-1, 1)
        stacked = np.vstack([v.toarray() for v in vectors])
        profile = (stacked * weights).sum(axis=0) / weights.sum()
        return profile.reshape(1, -1)

    def recommend_for_user(self, user_id: int, interactions: pd.DataFrame,
                            top_n: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        profile = self.user_profile_vector(user_id, interactions)
        if profile is None:
            return pd.DataFrame(columns=["product_id", "content_score"])
        sims = cosine_similarity(profile, self.tfidf_matrix).flatten()
        seen = set(interactions.loc[interactions["user_id"] == user_id, "product_id"]) if exclude_seen else set()
        order = np.argsort(-sims)
        results = []
        for i in order:
            pid = self.idx_to_product_id[i]
            if pid in seen:
                continue
            results.append((pid, float(sims[i])))
            if len(results) >= top_n:
                break
        return pd.DataFrame(results, columns=["product_id", "content_score"])
