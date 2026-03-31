"""
train.py
Trains one XGBoost binary classifier per Tox21 assay.
Handles class imbalance via scale_pos_weight.
Saves models to disk.
"""

import os
import pickle
import numpy as np
import logging
from xgboost import XGBClassifier
from src.preprocess import ASSAY_COLS, load_and_fingerprint, get_assay_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models"


def get_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight = negatives / positives for imbalance handling."""
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def train_model(X_train: np.ndarray, y_train: np.ndarray, scale_pos_weight: float) -> XGBClassifier:
    """Train a single XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_all(csv_path: str) -> dict:
    """
    Full training pipeline for all 12 assays.
    Returns a dict: {assay_name: {'model': ..., 'X_test': ..., 'y_test': ...}}
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Loading and fingerprinting dataset...")
    X, labels, _ = load_and_fingerprint(csv_path)

    results = {}

    for assay in ASSAY_COLS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {assay}")

        X_train, X_test, y_train, y_test = get_assay_split(X, labels, assay)
        spw = get_scale_pos_weight(y_train)
        logger.info(f"[{assay}] scale_pos_weight = {spw:.2f}")

        model = train_model(X_train, y_train, spw)

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{assay.replace('-', '_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[{assay}] Model saved → {model_path}")

        results[assay] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
        }

    logger.info("\nAll 12 models trained successfully.")
    return results


if __name__ == "__main__":
    train_all("data/tox21.csv")
