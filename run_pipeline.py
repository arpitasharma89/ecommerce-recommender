"""
run_pipeline.py
-----------------
Runs the complete ML pipeline end-to-end, exactly matching the flow:

    Raw Dataset -> Data Cleaning -> EDA -> Feature Engineering ->
    Train/Test Split -> Content-Based Model -> Collaborative Filtering
    Model -> Hybrid Recommendation Engine -> Evaluation -> Model Saving

Run once before launching the Streamlit app:
    python run_pipeline.py
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import PROCESSED_DIR, MODELS_DIR, RAW_DIR, get_logger  # noqa: E402
import generate_dataset  # noqa: E402
import data_preprocessing  # noqa: E402
import feature_engineering  # noqa: E402
from content_based import ContentBasedRecommender  # noqa: E402
from collaborative_filtering import CollaborativeRecommender  # noqa: E402
from hybrid_recommender import HybridRecommender  # noqa: E402
from evaluation import evaluate_recommender, tune_hybrid_weights  # noqa: E402

logger = get_logger("run_pipeline")


def main():
    # 1. Raw dataset (generate only if missing, so re-runs are fast/stable)
    if not (RAW_DIR / "interactions.csv").exists():
        logger.info("STEP 1/9: Generating raw dataset...")
        generate_dataset.main()
    else:
        logger.info("STEP 1/9: Raw dataset already present, skipping generation.")

    # 2. Cleaning
    logger.info("STEP 2/9: Cleaning data...")
    products, users, interactions = data_preprocessing.run()

    # 3. EDA (see src/eda.py -> generates charts into data/processed/eda_charts/)
    logger.info("STEP 3/9: Running EDA (charts saved to data/processed/eda_charts/)...")
    import eda
    eda.run()

    # 4-5. Feature engineering + train/test split
    logger.info("STEP 4/9: Feature engineering + train/test split...")
    product_features, user_features, train, test = feature_engineering.run()

    # 6. Content-based model
    logger.info("STEP 5/9: Training content-based model...")
    content_model = ContentBasedRecommender(product_features)

    # 7. Collaborative filtering model
    logger.info("STEP 6/9: Training collaborative filtering model...")
    collab_model = CollaborativeRecommender(n_factors=30)
    collab_model.fit(train)

    # 8. Hybrid model + weight tuning
    logger.info("STEP 7/9: Tuning hybrid weights (alpha/beta) via grid search on NDCG@10...")
    eval_users = test["user_id"].unique().tolist()
    # cap number of users evaluated during tuning to keep runtime reasonable
    tuning_users = eval_users[:300] if len(eval_users) > 300 else eval_users
    weight_results, best_alpha, best_beta = tune_hybrid_weights(
        content_model, collab_model, product_features, user_features, train, test,
        tuning_users, k=10
    )
    weight_results.to_csv(PROCESSED_DIR / "hybrid_weight_tuning.csv", index=False)

    hybrid_model = HybridRecommender(
        content_model, collab_model, product_features, user_features, train,
        alpha=best_alpha, beta=best_beta
    )

    # 9. Evaluation & comparison table
    logger.info("STEP 8/9: Evaluating all three models...")
    eval_users_full = eval_users[:500] if len(eval_users) > 500 else eval_users

    content_metrics = evaluate_recommender(
        lambda uid, topn: content_model.recommend_for_user(uid, train, top_n=topn),
        eval_users_full, test, product_features, k=10, content_model=content_model
    )
    collab_metrics = evaluate_recommender(
        lambda uid, topn: collab_model.recommend_for_user(uid, train, top_n=topn),
        eval_users_full, test, product_features, k=10, content_model=content_model
    )
    hybrid_metrics = evaluate_recommender(
        lambda uid, topn: hybrid_model.recommend_products(uid, topn),
        eval_users_full, test, product_features, k=10, content_model=content_model
    )

    comparison = pd.DataFrame([
        {"Model": "Content-Based", **content_metrics},
        {"Model": "Collaborative", **collab_metrics},
        {"Model": "Hybrid", **hybrid_metrics},
    ])
    comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
    print("\n=== MODEL COMPARISON (K=10) ===")
    print(comparison.to_string(index=False))

    # 10. Save models
    logger.info("STEP 9/9: Saving models to models/ ...")
    joblib.dump(content_model, MODELS_DIR / "content_model.joblib")
    joblib.dump(collab_model, MODELS_DIR / "collab_model.joblib")
    joblib.dump({"alpha": best_alpha, "beta": best_beta}, MODELS_DIR / "hybrid_weights.joblib")
    product_features.to_csv(MODELS_DIR / "products_features.csv", index=False)
    user_features.to_csv(MODELS_DIR / "users_features.csv", index=False)
    train.to_csv(MODELS_DIR / "interactions_train.csv", index=False)

    logger.info("Pipeline complete. Models saved in models/, "
                "charts in data/processed/eda_charts/, "
                "comparison table in data/processed/model_comparison.csv")


if __name__ == "__main__":
    main()
