"""
app.py
Streamlit web interface for drug toxicity prediction.
Sections:
  1. Single molecule prediction (SMILES input → 12 assay probabilities + SHAP)
  2. Batch prediction (CSV upload → downloadable results)
  3. Dataset insights (pre-generated plots)
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import shap
import io
from PIL import Image

from src.preprocess import ASSAY_COLS, smiles_to_fingerprint

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeCure — Drug Toxicity Predictor",
    page_icon="🧬",
    layout="wide",
)

MODEL_DIR = "models"
PLOTS_DIR = "plots"

# ─── Load all 12 models ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for assay in ASSAY_COLS:
        path = os.path.join(MODEL_DIR, f"{assay.replace('-', '_')}.pkl")
        with open(path, "rb") as f:
            models[assay] = pickle.load(f)
    return models

models = load_models()

# ─── Helper: predict single fingerprint ─────────────────────────────────────
def predict_smiles(smiles: str):
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None, None
    fp_2d = fp.reshape(1, -1)
    probs = {}
    for assay, model in models.items():
        prob = model.predict_proba(fp_2d)[0][1]
        probs[assay] = round(float(prob), 4)
    return fp, probs

# ─── Helper: SHAP waterfall for one molecule ────────────────────────────────
def shap_waterfall(fp: np.ndarray, assay: str, top_n: int = 15):
    model = models[assay]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(fp.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]  # 1D
    top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    top_vals = sv[top_idx]
    top_names = [f"bit_{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_vals[::-1]]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (→ toxic  |  ← non-toxic)", fontsize=10)
    ax.set_title(f"Feature Contributions — {assay}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig

# ─── Helper: draw molecule ───────────────────────────────────────────────────
def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 250)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

# ─── Risk colour helper ──────────────────────────────────────────────────────
def risk_color(prob: float) -> str:
    if prob >= 0.5:
        return "#e74c3c"   # red
    elif prob >= 0.3:
        return "#f39c12"   # orange
    else:
        return "#2ecc71"   # green

def risk_label(prob: float) -> str:
    if prob >= 0.5:
        return "HIGH RISK"
    elif prob >= 0.3:
        return "MODERATE"
    else:
        return "LOW RISK"

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Haemoglobin.jpg/240px-Haemoglobin.jpg", use_column_width=True)
st.sidebar.title("🧬 CodeCure")
st.sidebar.markdown("**AI-Powered Drug Toxicity Predictor**")
st.sidebar.markdown("Predicts toxicity across **12 biological assays** using Morgan Fingerprints + XGBoost.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔬 Single Molecule", "📋 Batch Prediction", "📊 Dataset Insights"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Built for:** CodeCure AI Hackathon — SPIRIT 2026")
st.sidebar.markdown("**Dataset:** Tox21 (7,831 molecules)")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE MOLECULE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
if page == "🔬 Single Molecule":
    st.title("🔬 Single Molecule Toxicity Prediction")
    st.markdown("Enter a SMILES string to predict toxicity across all 12 Tox21 assays.")

    # Example molecules
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    examples = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "Bisphenol A": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
        "Aflatoxin B1": "O=c1oc2c(OC)cc3c(c2c2c1[C@@H]1[C@H](O1)C=C2)OCO3",
    }
    if col_ex1.button("Aspirin"):
        st.session_state["smiles_input"] = examples["Aspirin"]
    if col_ex2.button("Caffeine"):
        st.session_state["smiles_input"] = examples["Caffeine"]
    if col_ex3.button("Bisphenol A"):
        st.session_state["smiles_input"] = examples["Bisphenol A"]
    if col_ex4.button("Aflatoxin B1"):
        st.session_state["smiles_input"] = examples["Aflatoxin B1"]

    smiles = st.text_input(
        "SMILES String",
        value=st.session_state.get("smiles_input", "CC(=O)Oc1ccccc1C(=O)O"),
        placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
    )

    if st.button("🚀 Predict Toxicity", type="primary"):
        if not smiles.strip():
            st.error("Please enter a SMILES string.")
        else:
            with st.spinner("Computing predictions..."):
                fp, probs = predict_smiles(smiles.strip())

            if probs is None:
                st.error("❌ Invalid SMILES string. Please check your input.")
            else:
                # Layout: molecule structure | overall risk summary
                col_mol, col_summary = st.columns([1, 2])

                with col_mol:
                    st.subheader("Molecule Structure")
                    svg = draw_molecule(smiles)
                    if svg:
                        st.image(svg.encode(), use_column_width=True)
                    else:
                        st.write("Could not render structure.")

                with col_summary:
                    st.subheader("Toxicity Risk Overview")
                    high_risk = [a for a, p in probs.items() if p >= 0.5]
                    moderate = [a for a, p in probs.items() if 0.3 <= p < 0.5]
                    low_risk = [a for a, p in probs.items() if p < 0.3]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔴 High Risk Assays", len(high_risk))
                    c2.metric("🟡 Moderate Risk", len(moderate))
                    c3.metric("🟢 Low Risk", len(low_risk))

                    if high_risk:
                        st.error(f"**High risk detected in:** {', '.join(high_risk)}")
                    elif moderate:
                        st.warning(f"**Moderate risk in:** {', '.join(moderate)}")
                    else:
                        st.success("✅ Low toxicity risk across all assays")

                # Probability bars — all 12 assays
                st.subheader("Probability by Assay")
                cols = st.columns(3)
                for i, (assay, prob) in enumerate(probs.items()):
                    col = cols[i % 3]
                    with col:
                        color = risk_color(prob)
                        label = risk_label(prob)
                        col.markdown(f"""
                        <div style='border:1px solid #ddd; border-radius:8px; padding:10px; margin:4px 0;'>
                            <b>{assay}</b><br>
                            <div style='background:#eee; border-radius:4px; height:18px; margin:4px 0;'>
                                <div style='background:{color}; width:{prob*100:.0f}%; height:100%; border-radius:4px;'></div>
                            </div>
                            <span style='color:{color}; font-weight:bold;'>{prob:.1%}</span>
                            &nbsp;·&nbsp;<span style='font-size:0.85em'>{label}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # SHAP explanation
                st.subheader("🔍 Explain a Prediction (SHAP)")
                selected_assay = st.selectbox("Select assay to explain:", ASSAY_COLS)

                with st.spinner("Computing SHAP values..."):
                    fig_shap = shap_waterfall(fp, selected_assay)
                st.pyplot(fig_shap)
                plt.close(fig_shap)
                st.caption("Red bars = features pushing toward TOXIC; Green bars = features pushing toward NON-TOXIC.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Prediction":
    st.title("📋 Batch Molecule Prediction")
    st.markdown("Upload a CSV with a `smiles` column. Get predictions for all 12 assays.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_input = pd.read_csv(uploaded)
        if "smiles" not in df_input.columns:
            st.error("CSV must have a column named `smiles`.")
        else:
            st.info(f"Found {len(df_input)} molecules. Running predictions...")
            progress = st.progress(0)
            results = []
            for i, row in df_input.iterrows():
                smi = str(row["smiles"])
                _, probs = predict_smiles(smi)
                if probs is None:
                    probs = {a: None for a in ASSAY_COLS}
                entry = {"smiles": smi}
                entry.update(probs)
                results.append(entry)
                progress.progress((i + 1) / len(df_input))

            df_out = pd.DataFrame(results)
            st.success(f"✅ Predictions complete for {len(df_out)} molecules!")
            st.dataframe(df_out.head(20))

            csv_bytes = df_out.to_csv(index=False).encode()
            st.download_button("⬇️ Download Full Results (CSV)", csv_bytes, "toxicity_predictions.csv", "text/csv")
    else:
        st.markdown("""
        **Expected CSV format:**
        ```
        smiles
        CC(=O)Oc1ccccc1C(=O)O
        Cn1cnc2c1c(=O)n(c(=O)n2C)C
        ...
        ```
        """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Insights":
    st.title("📊 Dataset & Model Insights")

    # AUC scores
    st.subheader("Model Performance (AUC-ROC per Assay)")
    auc_path = os.path.join(PLOTS_DIR, "auc_scores.png")
    if os.path.exists(auc_path):
        st.image(auc_path, use_column_width=True)

    # Metrics table
    metrics_path = os.path.join(PLOTS_DIR, "metrics_summary.csv")
    if os.path.exists(metrics_path):
        df_m = pd.read_csv(metrics_path)
        st.subheader("Full Metrics Table")
        st.dataframe(df_m.set_index("Assay").style.background_gradient(cmap="RdYlGn", subset=["AUC-ROC"]))

    # Class distribution
    st.subheader("Class Imbalance per Assay")
    dist_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    if os.path.exists(dist_path):
        st.image(dist_path, use_column_width=True)
    st.caption("The severe imbalance (most molecules are non-toxic) is handled via XGBoost's scale_pos_weight parameter.")

    # Confusion matrices
    st.subheader("Confusion Matrices — All 12 Assays")
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_column_width=True)

    # SHAP heatmap
    st.subheader("Cross-Assay Feature Importance (SHAP Heatmap)")
    heatmap_path = os.path.join(PLOTS_DIR, "shap_heatmap.png")
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, use_column_width=True)
    st.caption("Rows = assays, Columns = top Morgan fingerprint bits. Brighter = more important for predicting toxicity in that assay.")

    # Per-assay SHAP plots
    st.subheader("Per-Assay SHAP Feature Importance")
    selected = st.selectbox("Select assay:", ASSAY_COLS)
    shap_path = os.path.join(PLOTS_DIR, f"shap_{selected.replace('-', '_')}.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_column_width=True)
