import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import torch
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
from tabpfn import TabPFNRegressor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="PymatgenData")

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Feature calculators (same as baseline)
# ------------------------------------------------------------
comp_feature_calculators = MultipleFeaturizer([
    cf.Stoichiometry(),
    cf.ElementProperty.from_preset("magpie", impute_nan=True),
    cf.ValenceOrbital(props=["avg"]),
    cf.IonProperty(fast=True)
])
density_feat = DensityFeatures()
sym_feat = GlobalSymmetryFeatures()
hetero_feat = StructuralHeterogeneity()
mindist_feat = MinimumRelativeDistances(flatten=True)
rdf_feat = RadialDistributionFunction(bin_size=0.1, cutoff=10.0)

# ------------------------------------------------------------
# Utility: simple featurization (use only once)
# ------------------------------------------------------------
def compute_all_features(df, target_col, cache_prefix="cached_features_fm"):
    X_path, y_path = f"{cache_prefix}_X.npy", f"{cache_prefix}_y.npy"
    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"✅ Loading cached features from {X_path}, {y_path}")
        X = np.load(X_path)
        y = np.load(y_path)
        return X, y

    print("⚙️ Computing features for FM (one-time) ...")
    feats, targets = [], []

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
            sym_fvec = np.array(sym_feat.featurize(s), dtype=np.float32)
            hetero_fvec = hetero_feat.featurize(s)
            mindist_fvec = np.array(mindist_feat.featurize(s), dtype=np.float32)
            rdf_fvec = np.array(rdf_feat.featurize(s), dtype=np.float32)
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
    print(f"✅ Saved cached FM features to {X_path}, {y_path}")
    return X, y

# ------------------------------------------------------------
# Train TabPFN Foundation Model
# ------------------------------------------------------------
def train_tasks(data_file="thermalconductivity_K_total3149.json",
                n_splits=5,
                cache_prefix="cached_features",
                length=-1):

    # Load dataset
    df = pd.read_json(data_file)
    if length > 0:
        df = df.iloc[:length].reset_index(drop=True)
    df["structure"] = df["structure"].apply(Structure.from_dict)
    target_col = df.columns[-1]
    print(f"Target column: {target_col}")

    # Compute or load cached features
    X, y = compute_all_features(df, target_col, cache_prefix)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold} (TabPFN FM) ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_dir = f"FM_{fold}"
        os.makedirs(fold_dir, exist_ok=True)

        # 3️⃣  Initialize & train FM (TabPFN)
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        print(f"Training TabPFN on fold {fold} ...")
        model.fit(X_train.tolist(), y_train.tolist())

        # 4️⃣  Save model
        model_path = os.path.join(fold_dir, "tabpfn_model.pt")
        torch.save(model, model_path)
        print(f"✅ Saved model to {model_path}")

        # 5️⃣  Evaluate
        preds = model.predict(X_test.tolist())
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2s.append(r2)
        print(f"[Fold {fold}] MAE={mae:.6f}, R²={r2:.6f}")

        pd.DataFrame({"target": y_test, "prediction": preds}).to_csv(
            os.path.join(fold_dir, f"tabpfn_predictions_fold{fold}.csv"), index=False
        )

        torch.cuda.empty_cache()

    # Summary
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)
    print("\n=== Overall Results (FM TabPFN) ===")
    print(f"MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")

    summary_lines = [
        "Per-fold FM results:",
        *[f"Fold {i}: MAE={m:.6f}, R²={r:.6f}" for i, (m, r) in enumerate(zip(maes, r2s))],
        "",
        f"Final MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}",
        f"Final R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}",
    ]
    with open("FM_results_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("📄 Saved FM_results_summary.txt")

    return maes, r2s


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print(os.getcwd())
    train_tasks("../../farm/thermalconductivity_K_total3149.json", n_splits=5)
