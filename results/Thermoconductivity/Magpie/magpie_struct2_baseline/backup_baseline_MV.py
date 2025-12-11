import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump

from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    StructuralHeterogeneity,
    MinimumRelativeDistances,
    RadialDistributionFunction,
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="PymatgenData")

np.random.seed(42)

# ------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------
comp_feature_calculators = MultipleFeaturizer([
    cf.Stoichiometry(),
    cf.ElementProperty.from_preset("magpie", impute_nan=True),
    cf.ValenceOrbital(props=['avg']),
    cf.IonProperty(fast=True)
])

density_feat = DensityFeatures()
sym_feat = GlobalSymmetryFeatures()
hetero_feat = StructuralHeterogeneity()
mindist_feat = MinimumRelativeDistances(flatten=True)
rdf_feat = RadialDistributionFunction(bin_size=0.1, cutoff=10.0)

from pymatgen.core.periodic_table import Element
def safe_to_float(val):
    """Convert to float if possible; if string is element symbol, use atomic number."""
    try:
        return float(val)
    except Exception:
        try:
            # Handle element symbol or oxidation like 'As' or 'As2-'
            symbol = ''.join([c for c in str(val) if c.isalpha()])  # extract letters
            if symbol:
                return float(Element(symbol).Z)
            else:
                return 0.0
        except Exception:
            return 0.0

import re   
from collections.abc import Iterable
  
def sanitize_feature_vector(vec):
    """
    Safely flatten and convert a featurizer output (which may contain strings,
    tuples, lists, or nested arrays) to numeric floats.

    Rules:
      - Numeric values -> keep as float
      - Strings like 'As2-', 'Cd2+' -> atomic number (33, 48)
      - Tuples/lists of species -> count their length
      - Anything else -> 0.0
    """
    flat = []

    def _flatten(x):
        # Recursively flatten lists/tuples
        if isinstance(x, (list, tuple)):
            # If tuple/list contains only element strings like ('As2-', 'As2-'), take its count
            if all(isinstance(i, (str,)) for i in x):
                flat.append(len(x))
            else:
                for i in x:
                    _flatten(i)
        else:
            flat.append(x)

    _flatten(vec)

    cleaned = []
    for v in flat:
        # Keep numeric values
        if isinstance(v, (int, float, np.integer, np.floating)):
            cleaned.append(float(v))
            continue

        # Try to convert element symbol strings
        s = str(v).strip()
        if s in ("nan", "None", "", "NaN"):
            cleaned.append(0.0)
            continue

        # Extract element letters (As2- -> As)
        letters = "".join(re.findall(r"[A-Za-z]+", s))
        try:
            if letters:
                cleaned.append(float(Element(letters).Z))
            else:
                cleaned.append(0.0)
        except Exception:
            cleaned.append(0.0)

    return np.array(cleaned, dtype=np.float32)

def fill_nan_vector(vec, fill_value=0.0):
    arr = np.array(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def safe_symmetry_features(structure):
    """Return only numeric symmetry features, ignoring categorical strings."""
    vals = sym_feat.featurize(structure)
    labels = sym_feat.feature_labels()
    numeric_vals = []
    for name, val in zip(labels, vals):
        if isinstance(val, (int, float, np.integer, np.floating)):
            numeric_vals.append(float(val))
    return np.array(numeric_vals, dtype=np.float32)
    
def flatten_vector(vec, fixed_len=None):
    """Ensure 1-D numeric vector; zero-pad or flatten irregular results."""
    arr = np.atleast_1d(np.array(vec, dtype=np.float32).ravel())
    if fixed_len and arr.size != fixed_len:
        tmp = np.zeros(fixed_len, dtype=np.float32)
        n = min(fixed_len, arr.size)
        tmp[:n] = arr[:n]
        arr = tmp
    return arr

def featurize_df(datdf, target_col):
    """Featurize dataset using Magpie + structure descriptors, handling non-float values."""
    feats, targets = [], []

    for jj, j in tqdm(datdf.iterrows(), total=len(datdf), desc="Featurizing"):
        
            s = j.structure
            comp = s.composition

            # --- Composition features ---
            comp_fvec = flatten_vector(comp_feature_calculators.featurize(comp))

            # --- Structure features ---
            dens_fvec = flatten_vector(density_feat.featurize(s))
            sym_fvec = flatten_vector(safe_symmetry_features(s))
            hetero_fvec = flatten_vector(hetero_feat.featurize(s))
            #print(mindist_feat.featurize(s))
            try:
                mindist_fvec = sanitize_feature_vector(mindist_feat.featurize(s))
            except Exception as e:
                print(f"Warning: mindist_feat failed for structure {jj}: {e}")
                print( mindist_feat.featurize(s))
                exit
                #mindist_fvec = np.zeros(10, dtype=np.float32)  # fallback to zeros
            rdf_fvec = flatten_vector(rdf_feat.featurize(s), fixed_len=100)  # 100 bins

            # Combine everything
            all_parts = [comp_fvec, dens_fvec, sym_fvec, hetero_fvec, rdf_fvec, mindist_fvec]
            fvec = np.concatenate(all_parts, axis=0)

            # Convert all entries safely to float (and replace strings)
            fvec = np.array([safe_to_float(x) for x in fvec], dtype=np.float32)

            # 🧾 Debug: print feature types for first few structures
            if jj < 3:
                print(f"\n--- Structure {jj} ---")
                #for i, v in enumerate(fvec):
                    #print(f"Feature {i}: {v} ({type(v).__name__})")

            feats.append(fvec)
            targets.append(j[target_col])

        
            
            

    return np.stack(feats), np.array(targets, dtype=np.float32)

def train_tasks(
    data_file="thermalconductivity_K_total3149.json",
    n_splits=5,
    model_type="rf",
    length=-1,
):
    """Train Random Forest or MLP with composition + structure-aware features."""

    # 1️⃣ Load dataset
    df = pd.read_json(data_file)
    if length > 0:
        df = df.iloc[:length].reset_index(drop=True)
    df["structure"] = df["structure"].apply(Structure.from_dict)

    target_col = df.columns[-1]
    print(f"Target column: {target_col}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n=== Fold {fold} ({model_type.upper()}) ===")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        fold_name = f"{model_type.upper()}_{fold}"
        os.makedirs(fold_name, exist_ok=True)

        # ⚙️ Fit MinimumRelativeDistances on training structures
        print("Fitting MinimumRelativeDistances on training data ...")
        try:
            mindist_feat.fit([s for s in train_df["structure"]])
        except Exception as e:
            print("⚠️ Warning: fit failed (some structures invalid) -> skipping:", e)

        # 2️⃣ Featurize train/test splits
        X_train, y_train = featurize_df(train_df, target_col)
        X_test, y_test = featurize_df(test_df, target_col)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 3️⃣ Choose model
        if model_type.lower() == "rf":
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            )
        elif model_type.lower() == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # 4️⃣ Train
        print(f"Training {model_type.upper()} model on fold {fold} ...")
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(fold_name, f"{model_type}_model_fold{fold}.joblib")
        dump(model, model_path)
        print(f"✅ Saved model to {model_path}")

        # 5️⃣ Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2s.append(r2)
        print(f"[Fold {fold}] MAE = {mae:.6f}, R² = {r2:.6f}")

        # Save predictions
        pd.DataFrame({"target": y_test, "prediction": preds}).to_csv(
            os.path.join(fold_name, f"{model_type}_predictions_fold{fold}.csv"), index=False
        )

    # 6️⃣ Summary
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)

    print("\n=== Overall Results ===")
    print(f"{model_type.upper()} | MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"{model_type.upper()} | R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")

    # Save summary
    result_lines = [
        f"Per-fold results for {model_type.upper()} model:",
        *[f"Fold {i}: MAE={m:.6f}, R²={r:.6f}" for i, (m, r) in enumerate(zip(maes, r2s))],
        "",
        f"Final MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}",
        f"Final R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}",
    ]
    with open(f"{model_type}_results_summary.txt", "w") as f:
        f.write("\n".join(result_lines))

    print(f"\nSaved {model_type}_results_summary.txt with per-fold and overall scores.")

    return maes, r2s


if __name__ == "__main__":
    print(os.getcwd())
    # Run for Random Forest
    train_tasks(
        data_file="../../farm/thermalconductivity_K_total3149.json",
        model_type="rf",
        n_splits=5,
    )
    # Run for MLP
    train_tasks(
        data_file="../../farm/thermalconductivity_K_total3149.json",
        model_type="mlp",
        n_splits=5,
    )
