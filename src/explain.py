"""
explain.py
Generates SHAP-based feature importance plots for all 12 models.
Produces:
  - Global SHAP summary bar plots per assay
  - Combined top-features heatmap across all assays
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from src.preprocess import ASSAY_COLS, load_and_fingerprint, get_assay_split
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
PLOTS_DIR = "plots"
SHAP_SAMPLE = 300   # use a subset for speed (SHAP on full dataset is slow)


def load_model(assay: str):
    path = os.path.join(MODEL_DIR, f"{assay.replace('-', '_')}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def explain_all(csv_path: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    logger.info("Loading dataset for SHAP analysis...")
    X, labels, _ = load_and_fingerprint(csv_path)

    feature_names = [f"bit_{i}" for i in range(X.shape[1])]

    # Store mean |SHAP| per feature per assay for heatmap
    top_k = 20
    heatmap_data = {}

    for assay in ASSAY_COLS:
        logger.info(f"[{assay}] Computing SHAP values...")

        X_train, X_test, y_train, y_test = get_assay_split(X, labels, assay)
        model = load_model(assay)

        # Sample for speed
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
        X_sample = X_test[idx]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary XGBoost, shap_values is 2D (samples x features)
        if isinstance(shap_values, list):
            sv = shap_values[1]   # positive class
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        top_idx       = np.argsort(mean_abs_shap)[::-1][:top_k]
        top_vals      = mean_abs_shap[top_idx]
        top_names     = [feature_names[i] for i in top_idx]

        heatmap_data[assay] = pd.Series(mean_abs_shap, index=feature_names)

        # Per-assay SHAP bar plot
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_k))
        ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(f"Top {top_k} Important Molecular Features\n{assay}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Fingerprint Bit")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"shap_{assay.replace('-', '_')}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[{assay}] SHAP plot saved → {path}")

    # Cross-assay heatmap: top 30 most important bits overall
    logger.info("Generating cross-assay SHAP heatmap...")
    heatmap_df = pd.DataFrame(heatmap_data)

    # Pick top 30 bits by max SHAP across any assay
    top_bits = heatmap_df.max(axis=1).nlargest(30).index
    heatmap_subset = heatmap_df.loc[top_bits].T

    fig_h, ax_h = plt.subplots(figsize=(18, 8))
    sns_data = heatmap_subset.values
    im = ax_h.imshow(sns_data, aspect="auto", cmap="YlOrRd")
    ax_h.set_xticks(range(len(top_bits)))
    ax_h.set_xticklabels(top_bits, rotation=90, fontsize=8)
    ax_h.set_yticks(range(len(ASSAY_COLS)))
    ax_h.set_yticklabels(ASSAY_COLS, fontsize=10)
    plt.colorbar(im, ax=ax_h, label="Mean |SHAP|")
    ax_h.set_title("Cross-Assay Feature Importance Heatmap\n(Top 30 Morgan Fingerprint Bits)", 
                   fontsize=13, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, "shap_heatmap.png")
    fig_h.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    logger.info(f"SHAP heatmap saved → {heatmap_path}")

    logger.info("SHAP analysis complete.")


if __name__ == "__main__":
    explain_all("data/tox21.csv")
