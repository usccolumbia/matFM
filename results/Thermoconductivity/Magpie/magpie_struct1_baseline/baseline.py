import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump  # for saving models efficiently

from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ------------------------------------------------------------
# Featurizer setup: Magpie + Density + Symmetry
# ------------------------------------------------------------
comp_feature_calculators = MultipleFeaturizer([
    cf.Stoichiometry(),
    cf.ElementProperty.from_preset("magpie"),
    cf.ValenceOrbital(props=['avg']),
    cf.IonProperty(fast=True)
])

density_feat = DensityFeatures()
sym_feat = GlobalSymmetryFeatures()


def fill_nan_vector(vec, fill_value=0.0):
    arr = np.array(vec, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr

def safe_symmetry_features(structure):
    """Return only numeric symmetry features, ignoring strings like 'tetragonal'."""
    vals = sym_feat.featurize(structure)
    labels = sym_feat.feature_labels()
    numeric_vals = []
    for name, val in zip(labels, vals):
        # Keep only numeric (int or float) values
        if isinstance(val, (int, float, np.integer, np.floating)):
            numeric_vals.append(float(val))
        # Skip string/categorical features
    return np.array(numeric_vals, dtype=np.float32)


def featurize_df(datdf, target_col):
    """Featurize dataset using Magpie (composition) + Density + Symmetry."""
    feats, targets = [], []

    for jj, j in tqdm(datdf.iterrows(), total=len(datdf), desc="Featurizing"):
        try:
            s = j.structure
            comp = s.composition

            # --- Magpie features ---
            comp_fvec = comp_feature_calculators.featurize(comp)

            # --- Structure features ---
            dens_fvec = density_feat.featurize(s)
            sym_fvec = safe_symmetry_features(s)

            # Combine everything
            fvec = np.concatenate([comp_fvec, dens_fvec, sym_fvec], axis=0)
            fvec = fill_nan_vector(fvec)

            feats.append(fvec)
            targets.append(j[target_col])
        except Exception as e:
            print(f"Skipping {jj}: {e}")
            continue

    return np.stack(feats), np.array(targets, dtype=np.float32)


def train_tasks(data_file="thermalconductivity_K_total3149.json",
                n_splits=5,
                model_type="rf",   # or "mlp"
                length=-1):
    """Train Random Forest or MLP on thermo dataset using Magpie + structure features."""
    
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
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        fold_name = f"{model_type.upper()}_{fold}"
        os.makedirs(fold_name, exist_ok=True)

        # 2️⃣ Compute features
        X_train, y_train = featurize_df(train_df, target_col)
        X_test,  y_test  = featurize_df(test_df, target_col)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 3️⃣ Choose model
        if model_type.lower() == "rf":
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                random_state=42
            )
        elif model_type.lower() == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=1e-3,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # 4️⃣ Train model
        print(f"Training {model_type.upper()} model on fold {fold}...")
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(fold_name, f"{model_type}_model_fold{fold}.joblib")
        dump(model, model_path)
        print(f"✅ Saved model to {model_path}")

        # 5️⃣ Predict and evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2s.append(r2)

        # 6️⃣ Save predictions
        pd.DataFrame({
            "target": y_test,
            "prediction": preds
        }).to_csv(os.path.join(fold_name, f"{model_type}_predictions_fold{fold}.csv"), index=False)

        print(f"[Fold {fold}] MAE = {mae:.6f}, R² = {r2:.6f}")

    # 7️⃣ Summary
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)

    print("\n=== Overall Results ===")
    print(f"{model_type.upper()} | MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"{model_type.upper()} | R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")

    # 8️⃣ Save summary
    result_text = []
    result_text.append(f"Per-fold results for {model_type.upper()} model:\n")
    for i, (m, r) in enumerate(zip(maes, r2s)):
        result_text.append(f"Fold {i}: MAE = {m:.6f}, R² = {r:.6f}")
    result_text.append("\n")
    result_text.append(f"Final MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    result_text.append(f"Final R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")

    with open(f"{model_type}_results_summary.txt", "w") as f:
        f.write("\n".join(result_text))

    print(f"\nSaved {model_type}_results_summary.txt with per-fold and overall scores.")

    return maes, r2s


if __name__ == "__main__":
    print(os.getcwd())
    # Run for Random Forest
    train_tasks(data_file="../../farm/thermalconductivity_K_total3149.json", model_type="rf", n_splits=5)
    # Run for MLP
    train_tasks(data_file="../../farm/thermalconductivity_K_total3149.json", model_type="mlp", n_splits=5)
