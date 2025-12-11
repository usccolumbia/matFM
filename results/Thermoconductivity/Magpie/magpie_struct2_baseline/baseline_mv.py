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
import re
from collections.abc import Iterable


# -----------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------
def sanitize_feature_vector(vec):
    """Flatten nested outputs, convert species strings to atomic numbers, count tuples."""
    flat = []

    def _flatten(x):
        if isinstance(x, (list, tuple)):
            if all(isinstance(i, str) for i in x):
                flat.append(len(x))
            else:
                for i in x:
                    _flatten(i)
        else:
            flat.append(x)
    _flatten(vec)

    cleaned = []
    for v in flat:
        if isinstance(v, (int, float, np.integer, np.floating)):
            cleaned.append(float(v))
            continue
        s = str(v).strip()
        if s in ("nan", "None", "", "NaN"):
            cleaned.append(0.0)
            continue
        letters = "".join(re.findall(r"[A-Za-z]+", s))
        try:
            cleaned.append(float(Element(letters).Z) if letters else 0.0)
        except Exception:
            cleaned.append(0.0)
    return np.array(cleaned, dtype=np.float32)


def safe_symmetry_features(structure):
    vals = sym_feat.featurize(structure)
    labels = sym_feat.feature_labels()
    numeric_vals = []
    for name, val in zip(labels, vals):
        if isinstance(val, (int, float, np.integer, np.floating)):
            numeric_vals.append(float(val))
    return np.array(numeric_vals, dtype=np.float32)


# -----------------------------------------------------------------
# One-time featurization
# -----------------------------------------------------------------
def compute_all_features(df, target_col, cache_prefix="cached_features"):
    """Compute and cache all features and targets once."""
    X_path, y_path = f"{cache_prefix}_X.npy", f"{cache_prefix}_y.npy"
    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"✅ Loading cached features: {X_path}, {y_path}")
        X = np.load(X_path)
        y = np.load(y_path)
        return X, y

    print("⚙️ Computing features from scratch ...")
    feats, targets = [], []

    # Fit MinimumRelativeDistances once on full dataset
    try:
        mindist_feat.fit([s for s in df["structure"]])
    except Exception as e:
        print("⚠️ mindist_feat.fit warning:", e)

    for jj, j in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        try:
            s = j.structure
            comp = s.composition

            comp_fvec = comp_feature_calculators.featurize(comp)
            dens_fvec = density_feat.featurize(s)
            sym_fvec = safe_symmetry_features(s)
            hetero_fvec = hetero_feat.featurize(s)
            mindist_fvec = sanitize_feature_vector(mindist_feat.featurize(s))
            rdf_fvec = rdf_feat.featurize(s)

            fvec = np.concatenate(
                [comp_fvec, dens_fvec, sym_fvec, hetero_fvec, mindist_fvec, rdf_fvec]
            ).astype(np.float32)
            feats.append(fvec)
            targets.append(j[target_col])
        except Exception as e:
            print(f"Skipping {jj}: {e}")
            continue

    X = np.vstack(feats)
    y = np.array(targets, dtype=np.float32)

    np.save(X_path, X)
    np.save(y_path, y)
    print(f"✅ Saved cached features to {X_path} and {y_path}")
    return X, y


# -----------------------------------------------------------------
# Training
# -----------------------------------------------------------------
def train_tasks(data_file="thermalconductivity_K_total3149.json",
                n_splits=5,
                model_type="rf",
                length=-1):

    # Load dataset
    df = pd.read_json(data_file)
    if length > 0:
        df = df.iloc[:length].reset_index(drop=True)
    df["structure"] = df["structure"].apply(Structure.from_dict)
    target_col = df.columns[-1]
    print(f"Target column: {target_col}")

    # Compute or load cached features
    X, y = compute_all_features(df, target_col)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold} ({model_type.upper()}) ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # NaN handling
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        
        fold_name = f"{model_type.upper()}_{fold}"
        os.makedirs(fold_name, exist_ok=True)

        # Choose model
        if model_type.lower() == "rf":
            model = RandomForestRegressor(
                n_estimators=500, max_depth=None, n_jobs=-1, random_state=42
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

        print(f"Training {model_type.upper()} model on fold {fold} ...")
        model.fit(X_train, y_train)
        dump(model, os.path.join(fold_name, f"{model_type}_model_fold{fold}.joblib"))
        print(f"✅ Saved model to {fold_name}/{model_type}_model_fold{fold}.joblib")

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2s.append(r2)
        print(f"[Fold {fold}] MAE={mae:.6f}, R²={r2:.6f}")

        pd.DataFrame({"target": y_test, "prediction": preds}).to_csv(
            os.path.join(fold_name, f"{model_type}_predictions_fold{fold}.csv"),
            index=False,
        )

    # Summary
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)
    print("\n=== Overall Results ===")
    print(f"{model_type.upper()} | MAE mean±std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"{model_type.upper()} | R² mean±std: {r2_mean:.6f} ± {r2_std:.6f}")

    result_lines = [
        f"Per-fold results for {model_type.upper()} model:",
        *[f"Fold {i}: MAE={m:.6f}, R²={r:.6f}" for i, (m, r) in enumerate(zip(maes, r2s))],
        "",
        f"Final MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}",
        f"Final R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}",
    ]
    with open(f"{model_type}_results_summary.txt", "w") as f:
        f.write("\n".join(result_lines))
    print(f"📄 Saved {model_type}_results_summary.txt")
    return maes, r2s


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    print(os.getcwd())
    # Random Forest baseline
    train_tasks("../../farm/thermalconductivity_K_total3149.json", model_type="rf", n_splits=5)
    # MLP baseline
    train_tasks("../../farm/thermalconductivity_K_total3149.json", model_type="mlp", n_splits=5)
