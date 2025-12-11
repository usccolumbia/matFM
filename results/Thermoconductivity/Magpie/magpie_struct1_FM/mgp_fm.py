import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import torch
from tabpfn import TabPFNRegressor  # ✅ 使用 TabPFN
from joblib import dump

from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ------------------------------------------------------------
# Featurizer setup
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
    vals = sym_feat.featurize(structure)
    labels = sym_feat.feature_labels()
    numeric_vals = []
    for name, val in zip(labels, vals):
        if isinstance(val, (int, float, np.integer, np.floating)):
            numeric_vals.append(float(val))
    return np.array(numeric_vals, dtype=np.float32)

def featurize_df(datdf, target_col):
    feats, targets = [], []
    for jj, j in tqdm(datdf.iterrows(), total=len(datdf), desc="Featurizing"):
        try:
            s = j.structure
            comp = s.composition
            comp_fvec = comp_feature_calculators.featurize(comp)
            dens_fvec = density_feat.featurize(s)
            sym_fvec = safe_symmetry_features(s)
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
                length=-1):
    """使用 TabPFN 进行回归任务"""
    
    # 检查可用设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1️⃣ 读取数据
    df = pd.read_json(data_file)
    if length > 0:
        df = df.iloc[:length].reset_index(drop=True)
    df["structure"] = df["structure"].apply(Structure.from_dict)

    target_col = df.columns[-1]
    print(f"Target column: {target_col}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n=== Fold {fold} (TabPFN) ===")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        fold_dir = f"TabPFN_{fold}"
        os.makedirs(fold_dir, exist_ok=True)

        # 2️⃣ 特征提取
        X_train, y_train = featurize_df(train_df, target_col)
        X_test,  y_test  = featurize_df(test_df, target_col)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 3️⃣ 初始化并训练 TabPFN
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        print(f"Training TabPFN on fold {fold}...")
        model.fit(X_train.tolist(), y_train.tolist())

        # 4️⃣ 保存模型
        model_path = os.path.join(fold_dir, "tabpfn_model.pt")
        torch.save(model, model_path)
        print(f"✅ Saved model to {model_path}")

        # 5️⃣ 预测与评估
        preds = model.predict(X_test.tolist())
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2s.append(r2)

        # 6️⃣ 保存预测结果
        pd.DataFrame({
            "target": y_test,
            "prediction": preds
        }).to_csv(os.path.join(fold_dir, f"tabpfn_predictions_fold{fold}.csv"), index=False)

        print(f"[Fold {fold}] MAE = {mae:.6f}, R² = {r2:.6f}")

    # 7️⃣ 汇总结果
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)

    print("\n=== Overall Results ===")
    print(f"TabPFN | MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"TabPFN | R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")

    with open("TabPFN_results_summary.txt", "w") as f:
        for i, (m, r) in enumerate(zip(maes, r2s)):
            f.write(f"Fold {i}: MAE = {m:.6f}, R² = {r:.6f}\n")
        f.write(f"\nFinal MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}\n")
        f.write(f"Final R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}\n")

    print("\nSaved TabPFN_results_summary.txt with per-fold and overall scores.")

    return maes, r2s


if __name__ == "__main__":
    train_tasks(data_file="../../farm/thermalconductivity_K_total3149.json", n_splits=5)
