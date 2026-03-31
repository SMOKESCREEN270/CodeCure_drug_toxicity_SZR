"""
evaluate.py
Evaluates all 12 trained models and produces:
  - Per-assay AUC-ROC, precision, recall, F1
  - Summary table (saved as CSV)
  - Confusion matrix plots (saved as PNG)
  - AUC-ROC bar chart (saved as PNG)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from src.preprocess import ASSAY_COLS, load_and_fingerprint, get_assay_split
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR  = "models"
PLOTS_DIR  = "plots"


def load_model(assay: str):
    path = os.path.join(MODEL_DIR, f"{assay.replace('-', '_')}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_all(csv_path: str) -> pd.DataFrame:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    logger.info("Loading dataset for evaluation...")
    X, labels, _ = load_and_fingerprint(csv_path)

    rows = []

    fig_cm, axes_cm = plt.subplots(3, 4, figsize=(20, 15))
    fig_cm.suptitle("Confusion Matrices — All 12 Tox21 Assays", fontsize=16, fontweight="bold")
    axes_flat = axes_cm.flatten()

    for i, assay in enumerate(ASSAY_COLS):
        _, X_test, _, y_test = get_assay_split(X, labels, assay)
        model = load_model(assay)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        auc  = roc_auc_score(y_test, y_pred_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        logger.info(f"[{assay}] AUC={auc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")
        rows.append({"Assay": assay, "AUC-ROC": round(auc, 3),
                     "Precision": round(prec, 3), "Recall": round(rec, 3), "F1": round(f1, 3)})

        # Confusion matrix subplot
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_flat[i],
                    xticklabels=["Non-toxic", "Toxic"],
                    yticklabels=["Non-toxic", "Toxic"])
        axes_flat[i].set_title(f"{assay}\nAUC={auc:.3f}", fontsize=10)
        axes_flat[i].set_xlabel("Predicted")
        axes_flat[i].set_ylabel("Actual")

    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cm)
    logger.info(f"Confusion matrices saved → {cm_path}")

    # Summary table
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(PLOTS_DIR, "metrics_summary.csv"), index=False)

    # AUC bar chart
    fig_auc, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2ecc71" if v >= 0.80 else "#f39c12" if v >= 0.70 else "#e74c3c"
              for v in summary_df["AUC-ROC"]]
    bars = ax.barh(summary_df["Assay"], summary_df["AUC-ROC"], color=colors, edgecolor="white")
    ax.axvline(0.8, color="green",  linestyle="--", linewidth=1.5, label="Good (0.80)")
    ax.axvline(0.7, color="orange", linestyle="--", linewidth=1.5, label="Fair (0.70)")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC-ROC Score", fontsize=12)
    ax.set_title("Model Performance per Tox21 Assay", fontsize=14, fontweight="bold")
    ax.legend()
    for bar, val in zip(bars, summary_df["AUC-ROC"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    auc_path = os.path.join(PLOTS_DIR, "auc_scores.png")
    fig_auc.savefig(auc_path, dpi=150, bbox_inches="tight")
    plt.close(fig_auc)
    logger.info(f"AUC chart saved → {auc_path}")

    # Class distribution plot
    _, labels_all, _ = load_and_fingerprint(csv_path)
    fig_dist, ax2 = plt.subplots(figsize=(14, 5))
    pos_counts = labels_all[ASSAY_COLS].sum()
    neg_counts = labels_all[ASSAY_COLS].notna().sum() - pos_counts
    x = np.arange(len(ASSAY_COLS))
    width = 0.35
    ax2.bar(x - width/2, neg_counts, width, label="Non-toxic (0)", color="#3498db", alpha=0.85)
    ax2.bar(x + width/2, pos_counts, width, label="Toxic (1)",     color="#e74c3c", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ASSAY_COLS, rotation=45, ha="right")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Class Distribution per Assay (showing severe imbalance)", fontsize=13, fontweight="bold")
    ax2.legend()
    plt.tight_layout()
    dist_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    fig_dist.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig_dist)
    logger.info(f"Class distribution saved → {dist_path}")

    print("\n" + "="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    return summary_df


if __name__ == "__main__":
    evaluate_all("data/tox21.csv")
