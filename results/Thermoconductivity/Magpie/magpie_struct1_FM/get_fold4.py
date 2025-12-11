import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
import warnings

warnings.filterwarnings("ignore")

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
    numeric_vals, numeric_labels = [], []
    for name, val in zip(labels, vals):
        if isinstance(val, (int, float, np.integer, np.floating)):
            numeric_vals.append(float(val))
            numeric_labels.append(name)
    return np.array(numeric_vals, dtype=np.float32), numeric_labels


def featurize_single(structure):
    """Featurize a single structure and return (vector, feature_labels)."""
    s = structure
    comp = s.composition
    comp_fvec = comp_feature_calculators.featurize(comp)
    dens_fvec = density_feat.featurize(s)
    sym_fvec, sym_labels = safe_symmetry_features(s)

    fvec = np.concatenate([comp_fvec, dens_fvec, sym_fvec], axis=0)
    fvec = fill_nan_vector(fvec)
    feature_labels = (
        comp_feature_calculators.feature_labels()
        + density_feat.feature_labels()
        + sym_labels
    )
    return fvec, feature_labels


def featurize_df(datdf, target_col):
    feats, targets = [], []
    feature_labels = None

    for jj, j in tqdm(datdf.iterrows(), total=len(datdf), desc="Featurizing"):
        try:
            fvec, labels = featurize_single(j.structure)
            feats.append(fvec)
            targets.append(j[target_col])
            if feature_labels is None:
                feature_labels = labels  # only take once
        except Exception as e:
            print(f"Skipping {jj}: {e}")
            continue

    return np.stack(feats), np.array(targets, dtype=np.float32), feature_labels


def save_fold4_data(data_file="thermalconductivity_K_total3149.json", n_splits=5, length=-1):
    """Featurize the dataset and store fold 4 train/test sets with feature names."""
    df = pd.read_json(data_file)
    if length > 0:
        df = df.iloc[:length].reset_index(drop=True)
    df["structure"] = df["structure"].apply(Structure.from_dict)

    target_col = df.columns[-1]
    print(f"Target column: {target_col}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        if fold != 4:
            continue

        print(f"\n=== Processing Fold 4 ===")
        fold_dir = "TabPFN_fold4_data"
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        X_train, y_train, feature_labels = featurize_df(train_df, target_col)
        X_test, y_test, _ = featurize_df(test_df, target_col)

        # fill NaN
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        np.savez(os.path.join(fold_dir, "fold4_data.npz"),
                 X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                 feature_labels=np.array(feature_labels))

        pd.DataFrame(X_train, columns=feature_labels).assign(target=y_train)\
            .to_csv(os.path.join(fold_dir, "fold4_train_labeled.csv"), index=False)
        pd.DataFrame(X_test, columns=feature_labels).assign(target=y_test)\
            .to_csv(os.path.join(fold_dir, "fold4_test_labeled.csv"), index=False)

        with open(os.path.join(fold_dir, "feature_labels.txt"), "w") as f:
            f.write("\n".join(feature_labels))

        print(f"✅ Saved fold 4 data with {len(feature_labels)} feature labels.")
        break


if __name__ == "__main__":
    tqdm.pandas()
    save_fold4_data(data_file="../../farm/thermalconductivity_K_total3149.json", n_splits=5)
