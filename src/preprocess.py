"""
preprocess.py
Converts SMILES strings to Morgan Fingerprints (2048-bit vectors)
and prepares per-assay train/test splits.
"""

import numpy as np
import pandas as pd
import warnings
import logging as _logging
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# Suppress RDKit deprecation and chemistry warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ASSAY_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

FINGERPRINT_RADIUS = 2       # Morgan radius (standard)
FINGERPRINT_BITS   = 2048    # fingerprint vector length


def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Convert a SMILES string to a 2048-bit Morgan fingerprint numpy array.
    Returns None if the SMILES is invalid or empty."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
    arr = np.zeros((FINGERPRINT_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_and_fingerprint(csv_path: str) -> tuple[np.ndarray, pd.DataFrame, pd.Index]:
    """
    Load CSV, compute fingerprints for all valid SMILES.
    Returns:
        X       : (N, 2048) fingerprint matrix
        labels  : (N, 12) label DataFrame with NaNs preserved
        valid_idx: original index of rows that survived SMILES parsing
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Convert SMILES → fingerprints
    fps = []
    valid_mask = []
    for smi in df["smiles"]:
        fp = smiles_to_fingerprint(str(smi))
        if fp is not None:
            fps.append(fp)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    dropped = (~valid_mask).sum()
    if dropped:
        logger.warning(f"Dropped {dropped} rows with invalid SMILES")

    df_valid = df[valid_mask].reset_index(drop=True)
    X = np.array(fps, dtype=np.uint8)
    labels = df_valid[ASSAY_COLS]

    logger.info(f"Fingerprint matrix shape: {X.shape}")
    return X, labels, df_valid.index


def get_assay_split(X: np.ndarray, labels: pd.DataFrame, assay: str,
                    test_size: float = 0.2, random_state: int = 42):
    """
    For a given assay, drop NaN rows and return stratified train/test split.
    Returns X_train, X_test, y_train, y_test
    """
    y = labels[assay]
    mask = y.notna()
    X_sub = X[mask]
    y_sub = y[mask].astype(int).values

    pos = y_sub.sum()
    neg = len(y_sub) - pos
    logger.info(f"[{assay}] {len(y_sub)} samples | pos={pos} ({100*pos/len(y_sub):.1f}%) | neg={neg}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub,
        test_size=test_size,
        stratify=y_sub,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
