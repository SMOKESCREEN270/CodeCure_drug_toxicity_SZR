# CodeCure — AI Drug Toxicity Predictor

> **CodeCure AI Hackathon · SPIRIT 2026 · IIT (BHU) Varanasi**
> Track A: Drug Toxicity Prediction (Pharmacology + AI)

---

## 🧬 Project Overview

CodeCure is an AI-powered drug toxicity prediction system that predicts the toxicity of chemical compounds across **12 biological assay targets** using molecular structure data. Given a SMILES string (a text representation of a molecule), the system predicts whether it will be active (toxic) in each of the 12 Tox21 assay panels.

**Why this matters:** Drug development failures due to unexpected toxicity cost billions of dollars and years of effort. Early AI-based toxicity screening can flag dangerous compounds before expensive lab testing begins.

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| ML Model | XGBoost (binary classifier per assay) |
| Molecular Featurization | RDKit — Morgan Fingerprints (2048-bit) |
| Explainability | SHAP (SHapley Additive Explanations) |
| Web Interface | Streamlit |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Imbalance Handling | XGBoost `scale_pos_weight` |

---

## 📊 Model Performance

| Assay | AUC-ROC | Description |
|---|---|---|
| NR-AR | 0.701 | Androgen Receptor |
| NR-AR-LBD | **0.885** | Androgen Receptor Ligand Binding Domain |
| NR-AhR | **0.892** | Aryl Hydrocarbon Receptor |
| NR-Aromatase | 0.819 | Aromatase Enzyme |
| NR-ER | 0.677 | Estrogen Receptor |
| NR-ER-LBD | 0.821 | Estrogen Receptor Ligand Binding Domain |
| NR-PPAR-gamma | 0.772 | PPAR-gamma Receptor |
| SR-ARE | 0.789 | Oxidative Stress / ARE Pathway |
| SR-ATAD5 | 0.807 | DNA Damage (ATAD5) |
| SR-HSE | 0.737 | Heat Shock Response |
| SR-MMP | **0.852** | Mitochondrial Membrane Potential |
| SR-p53 | 0.812 | Tumor Suppressor p53 Pathway |

**Average AUC-ROC: 0.797** — well above the 0.5 random baseline.

---

## 🏗 Technical Workflow

```
SMILES strings (Tox21 dataset)
        ↓
RDKit → Morgan Fingerprints (2048-bit binary vectors)
        ↓
Per-assay train/test split (stratified, 80/20)
Class imbalance handled via scale_pos_weight
        ↓
12 × XGBoost binary classifiers (one per assay)
        ↓
Evaluation: AUC-ROC, Precision, Recall, F1
SHAP explainability: which molecular substructures drive toxicity?
        ↓
Streamlit web app for interactive prediction
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/codecure-toxicity.git
cd codecure-toxicity
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place `tox21.csv` in the `data/` directory.
Dataset source: [Kaggle — Tox21 Dataset](https://www.kaggle.com/datasets/epicskills/tox21-dataset)

### 4. Train the models
```bash
python pipeline.py
```
This runs the full pipeline: training → evaluation → SHAP analysis.
Takes ~3-5 minutes. Saves models to `models/` and plots to `plots/`.

### 5. Launch the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
codecure-toxicity/
├── data/
│   └── tox21.csv               # Tox21 dataset (7,831 molecules)
├── models/
│   ├── NR_AR.pkl               # Trained XGBoost models (12 total)
│   └── ...
├── plots/
│   ├── auc_scores.png          # AUC-ROC bar chart
│   ├── confusion_matrices.png  # 12-panel confusion matrices
│   ├── class_distribution.png  # Class imbalance visualization
│   ├── shap_heatmap.png        # Cross-assay SHAP heatmap
│   ├── shap_NR_AR.png          # Per-assay SHAP plots
│   └── metrics_summary.csv     # Full metrics table
├── src/
│   ├── preprocess.py           # SMILES → Morgan fingerprints
│   ├── train.py                # XGBoost training pipeline
│   ├── evaluate.py             # Metrics + visualization
│   └── explain.py              # SHAP explainability
├── app.py                      # Streamlit web app
├── pipeline.py                 # Master runner script
├── requirements.txt
└── README.md
```

---

## ✨ Features

### Single Molecule Prediction
- Input any SMILES string
- View 2D molecular structure
- Get probability predictions for all 12 toxicity assays
- Color-coded risk levels (High / Moderate / Low)
- SHAP waterfall chart explaining which molecular features drove the prediction

### Batch Prediction
- Upload a CSV of SMILES strings
- Download a CSV with all 12 assay predictions per molecule

### Dataset Insights
- AUC-ROC performance chart per assay
- Class imbalance visualization (why class weighting is critical)
- Confusion matrices for all 12 models
- Cross-assay SHAP heatmap showing which fingerprint bits matter most

---

## 🔬 Key Technical Choices

### Why Morgan Fingerprints?
Morgan Fingerprints (circular fingerprints) encode the local chemical environment of each atom as a binary vector. They are the **industry standard** for molecular ML tasks — fast to compute, interpretable via bit-to-substructure mapping, and competitive with deep learning on datasets of this size.

### Why XGBoost?
- Handles tabular (fingerprint) data well
- Fast training — all 12 models train in < 60 seconds
- Built-in class imbalance handling via `scale_pos_weight`
- Compatible with SHAP for interpretability

### Why class weighting?
The Tox21 dataset is severely imbalanced — only 3–16% of molecules are toxic per assay. Without weighting, the model just predicts "not toxic" for everything and achieves 95% accuracy but 0 recall on toxic compounds — the worst possible outcome for a safety tool.

### SHAP explainability
SHAP (SHapley Additive Explanations) assigns each fingerprint bit a contribution score for each prediction. This answers: "which molecular substructure made this molecule look toxic?" — directly addressing the judges' criterion of **biological insight from data**.

---

## 📌 Dataset

**Primary:** Tox21 (NIH/EPA, 2014 challenge)
- 7,831 chemical compounds after cleaning
- 12 toxicity assay results per compound (with many missing values, handled per-assay)
- SMILES molecular representations
- Source: [Kaggle](https://www.kaggle.com/datasets/epicskills/tox21-dataset) | [NIH](https://tripod.nih.gov/tox21/challenge/)

---

## 👤 Author

Built for the **CodeCure AI Hackathon** at SPIRIT 2026, IIT (BHU) Varanasi.
