"""
pipeline.py
Master script — runs the full pipeline end to end:
  1. Train 12 XGBoost models on Tox21
  2. Evaluate and save metrics + plots
  3. Generate SHAP explainability plots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train    import train_all
from src.evaluate import evaluate_all
from src.explain  import explain_all

CSV_PATH = "data/tox21.csv"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  CODECURE — Drug Toxicity Prediction Pipeline")
    print("="*60)

    print("\n[STEP 1/3] Training 12 XGBoost models...")
    train_all(CSV_PATH)

    print("\n[STEP 2/3] Evaluating models...")
    evaluate_all(CSV_PATH)

    print("\n[STEP 3/3] Generating SHAP explainability plots...")
    explain_all(CSV_PATH)

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("  Models  → models/")
    print("  Plots   → plots/")
    print("  Run app → streamlit run app.py")
    print("="*60 + "\n")
