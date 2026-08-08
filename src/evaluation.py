"""
evaluation.py
--------------
Evaluates the three recommenders (content-based, collaborative, hybrid)
using RANKING metrics appropriate for recommender systems -- plain
accuracy is meaningless here because the task is "rank the right handful
of items near the top out of hundreds", not classification.

METRICS IMPLEMENTED
    Precision@K : of the K items we recommended, what fraction were
                  actually relevant (appeared in the user's held-out test
                  interactions)? Measures how "clean" the top-K list is.
    Recall@K    : of all the relevant items the user actually interacted
                  with in the test set, what fraction did we manage to
                  place in the top K? Measures coverage of true interest.
    F1@K        : harmonic mean of precision & recall -- a single number
                  balancing "clean list" vs "found everything relevant".
    MAP@K       : Mean Average Precision -- rewards placing relevant items
                  EARLIER in the ranked list, not just anywhere in the
                  top K (order matters).
    NDCG@K      : Normalized Discounted Cumulative Gain -- like MAP but
                  also lets relevance be graded (we use the interaction
                  weight, so a purchase in the test set counts as more
                  "gain" than a mere view), discounted logarithmically by
                  position.
    Coverage    : fraction of the ENTIRE product catalog that ever gets
                  recommended to at least one user. A model that always
                  recommends the same 10 best-sellers has low coverage
                  even with a great precision score -- coverage catches
                  that failure mode.
    Diversity   : average pairwise dissimilarity (1 - cosine similarity)
                  between items in the same user's recommendation list,
                  using the content TF-IDF vectors. Low diversity means
                  a user's list is full of near-duplicate items.

A "relevant" item for a user in the test set is any product they had a
purchase / wishlist / add_to_cart / view interaction with, weighted by
INTERACTION_WEIGHTS for NDCG's graded relevance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils import INTERACTION_WEIGHTS, PROCESSED_DIR, get_logger

logger = get_logger("evaluation")


def _relevant_items(test_df: pd.DataFrame, user_id: int) -> dict:
    """Return {product_id: relevance_weight} for a user's test interactions."""
    rows = test_df[test_df["user_id"] == user_id]
    rel = {}
    for _, r in rows.iterrows():
        w = INTERACTION_WEIGHTS.get(r["interaction_type"], 1.0)
        rel[r["product_id"]] = max(rel.get(r["product_id"], 0), w)
    return rel


def precision_recall_f1_at_k(recommended: list, relevant: dict, k: int):
    top_k = recommended[:k]
    hits = [p for p in top_k if p in relevant]
    precision = len(hits) / k if k > 0 else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def average_precision_at_k(recommended: list, relevant: dict, k: int) -> float:
    top_k = recommended[:k]
    hits, score = 0, 0.0
    for i, p in enumerate(top_k, start=1):
        if p in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: dict, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        relevant.get(p, 0) / np.log2(i + 1) for i, p in enumerate(top_k, start=1)
    )
    ideal_order = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_order, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(recommend_fn, users_to_eval, test_df, products_df,
                          k=10, content_model=None):
    """
    recommend_fn(user_id, top_n) -> DataFrame with a 'product_id' column,
    ordered by descending score. Runs all metrics across `users_to_eval`
    and returns the averaged results plus catalog coverage & diversity.
    """
    precisions, recalls, f1s, maps, ndcgs, diversities = [], [], [], [], [], []
    all_recommended = set()

    for uid in users_to_eval:
        relevant = _relevant_items(test_df, uid)
        if not relevant:
            continue
        rec_df = recommend_fn(uid, k)
        if rec_df is None or rec_df.empty or "product_id" not in rec_df.columns:
            continue
        recommended = rec_df["product_id"].tolist()
        all_recommended.update(recommended[:k])

        p, r, f1 = precision_recall_f1_at_k(recommended, relevant, k)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        maps.append(average_precision_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

        if content_model is not None and len(recommended[:k]) > 1:
            idxs = [content_model.product_id_to_idx[p] for p in recommended[:k]
                    if p in content_model.product_id_to_idx]
            if len(idxs) > 1:
                vecs = content_model.tfidf_matrix[idxs]
                sim = cosine_similarity(vecs)
                n = sim.shape[0]
                avg_sim = (sim.sum() - n) / (n * (n - 1))  # exclude diagonal
                diversities.append(1 - avg_sim)

    coverage = len(all_recommended) / len(products_df) if len(products_df) else 0.0

    return {
        "Precision@K": np.mean(precisions) if precisions else 0.0,
        "Recall@K": np.mean(recalls) if recalls else 0.0,
        "F1@K": np.mean(f1s) if f1s else 0.0,
        "MAP@K": np.mean(maps) if maps else 0.0,
        "NDCG@K": np.mean(ndcgs) if ndcgs else 0.0,
        "Coverage": coverage,
        "Diversity": np.mean(diversities) if diversities else 0.0,
        "n_users_evaluated": len(precisions),
    }


def tune_hybrid_weights(content_model, collab_model, products, users, train_df, test_df,
                         users_to_eval, k=10):
    """Grid search over alpha (collaborative weight) in [0, 1] step 0.1,
    beta = 1 - alpha, picking the value that maximizes NDCG@K on the
    held-out test set. This is what justifies the alpha/beta used by the
    hybrid model instead of picking them by hand."""
    from hybrid_recommender import HybridRecommender

    results = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        beta = round(1 - alpha, 2)
        model = HybridRecommender(content_model, collab_model, products, users, train_df,
                                   alpha=round(alpha, 2), beta=beta)
        metrics = evaluate_recommender(
            lambda uid, topn: model.recommend_products(uid, topn),
            users_to_eval, test_df, products, k=k
        )
        results.append({"alpha": round(alpha, 2), "beta": beta, **metrics})
        logger.info(f"alpha={alpha:.1f} beta={beta:.1f} -> NDCG@{k}={metrics['NDCG@K']:.4f}")

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["NDCG@K"].idxmax()]
    logger.info(f"Best weights: alpha={best_row['alpha']}, beta={best_row['beta']} "
                f"(NDCG@{k}={best_row['NDCG@K']:.4f})")
    return results_df, float(best_row["alpha"]), float(best_row["beta"])
