import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import torch
from tabpfn import TabPFNRegressor
from pymatgen.core import Structure, Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")

# --- Magpie-like feature featurizer ---
feature_calculators = MultipleFeaturizer([
    cf.Stoichiometry(),
    cf.ElementProperty.from_preset("magpie"),
    cf.ValenceOrbital(props=['avg']),
    cf.IonProperty(fast=True)
])

def fill_nan_vector(vec, fill_value=0.0):
    arr = np.array(vec, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr


def train_tasks(data_file="thermalconductivity_K_total3149.json", n_splits=5, length=-1):
    """Train TabPFN on thermo dataset using Magpie features."""
    
    # 1️⃣ Load the dataset
    df = pd.read_json(data_file)
    if length > 0:
        df = df.iloc[:length].reset_index(drop=True)
    df["structure"] = df["structure"].apply(Structure.from_dict)

    target_col = df.columns[-1]  # assume last column is property (e.g. K_total)
    print(f"Target column: {target_col}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n=== Fold {fold} ===")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        fold_name = f"Thermo_{fold}"
        os.makedirs(fold_name, exist_ok=True)

        # 2️⃣ Compute Magpie features
        def featurize_df(datdf):
            feats, targets = [], []
            for jj, j in tqdm(datdf.iterrows(), total=len(datdf), desc="Featurizing"):
                try:
                    comp = j.structure.composition
                    fvec = feature_calculators.featurize(comp)
                    fvec = fill_nan_vector(fvec)
                    feats.append(fvec)
                    targets.append(j[target_col])
                except Exception as e:
                    print(f"Skipping {jj}: {e}")
                    continue
            return np.stack(feats), np.array(targets, dtype=np.float32)

        X_train, y_train = featurize_df(train_df)
        X_test,  y_test  = featurize_df(test_df)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 3️⃣ Train TabPFN
        model_path = os.path.join(fold_name, "tabpfn_model.pt")
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)

        print(f"Training TabPFN on fold {fold}...")
        model.fit(X_train.tolist(), y_train.tolist())
        torch.save(model, model_path)
        print(f"Saved model to {model_path}")

        # 4️⃣ Predict and evaluate
        preds = model.predict(X_test.tolist())
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2s.append(r2)

        # 5️⃣ Save predictions
        pd.DataFrame({
            "target": y_test,
            "prediction": preds
        }).to_csv(os.path.join(fold_name, f"tabpfn_predictions_fold{fold}.csv"), index=False)

        print(f"[Fold {fold}] MAE = {mae:.6f}, R² = {r2:.6f}")

        del model
        torch.cuda.empty_cache()

    # 6️⃣ Summary across folds
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)

    print("\n=== Overall Results ===")
    print(f"MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")
    
    # 7️⃣ Save to text file
    result_text = []
    result_text.append("Per-fold MAE and R² results:\n")
    for i, (m, r) in enumerate(zip(maes, r2s)):
        result_text.append(f"Fold {i}: MAE = {m:.6f}, R² = {r:.6f}")
    result_text.append("\n")
    result_text.append(f"Final MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    result_text.append(f"Final R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")

    with open("results_summary.txt", "w") as f:
        f.write("\n".join(result_text))

    print("\nSaved results_summary.txt with per-fold and overall scores.")

    return maes, r2s


if __name__ == "__main__":

    print(os.getcwd())
    train_tasks(data_file="../../farm/thermalconductivity_K_total3149.json", n_splits=5)
