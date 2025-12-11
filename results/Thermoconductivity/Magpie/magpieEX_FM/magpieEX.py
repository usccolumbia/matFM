import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from tabpfn import TabPFNRegressor
from pymatgen.core import Structure, Composition, Element
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# Feature calculators
# ===========================
feature_calculators = MultipleFeaturizer([
    cf.Stoichiometry(),
    cf.ElementProperty.from_preset("magpie"),
    cf.ValenceOrbital(props=['avg']),
    cf.IonProperty(fast=True)
])


# --- Cation–Anion contrast features ---
class CationAnionContrastFeaturizer:
    def __init__(self):
        self.lonepair_cations = {"Pb", "Bi", "Sn", "Tl"}
        self.lonepair_anions = {"Se", "Te", "I"}

    def _get_props(self, el: Element):
        return {
            "en": el.X,
            "radius": el.atomic_radius,
            "valence_s": el.full_electronic_structure.count(("s", 1)),
            "valence_p": el.full_electronic_structure.count(("p", 1)),
            "valence_d": el.full_electronic_structure.count(("d", 1)),
            "valence_f": el.full_electronic_structure.count(("f", 1)),
            "ionization": el.ionization_energy if el.ionization_energy else np.nan,
            "electron_affinity": el.electron_affinity if el.electron_affinity else np.nan,
            "polarizability": getattr(el, "polarizability", np.nan),
        }

    def _group_mean(self, props_list, key):
        vals = [p[key] for p in props_list if p[key] is not None and not np.isnan(p[key])]
        return np.mean(vals) if vals else np.nan

    def featurize(self, formula: str):
        comp = Composition(formula)
        try:
            oxi_states = comp.oxi_state_guesses(max_sites=-1)[0]
        except Exception:
            return [np.nan] * len(self.feature_labels())

        cations, anions = [], []
        cation_oxi, anion_oxi = [], []
        cation_lone, anion_lone = 0, 0

        for el, ox in oxi_states.items():
            el = Element(el)
            props = self._get_props(el)
            if ox > 0:
                cations.append(props)
                cation_oxi.append(ox)
                if el.symbol in self.lonepair_cations:
                    cation_lone = 1
            elif ox < 0:
                anions.append(props)
                anion_oxi.append(ox)
                if el.symbol in self.lonepair_anions:
                    anion_lone = 1

        if len(cations) == 0 or len(anions) == 0:
            return [np.nan] * len(self.feature_labels())

        feats = {}
        feats["delta_en"] = self._group_mean(anions, "en") - self._group_mean(cations, "en")
        feats["delta_radius"] = self._group_mean(anions, "radius") - self._group_mean(cations, "radius")
        for orb in ["valence_s", "valence_p", "valence_d", "valence_f"]:
            feats[f"delta_{orb}"] = self._group_mean(cations, orb) - self._group_mean(anions, orb)
        feats["delta_IE_EA"] = self._group_mean(cations, "ionization") - self._group_mean(anions, "electron_affinity")
        feats["delta_polarizability"] = self._group_mean(anions, "polarizability") - self._group_mean(cations, "polarizability")
        feats["bond_ionicity"] = 1 - np.exp(-0.25 * feats["delta_en"] ** 2) if not np.isnan(feats["delta_en"]) else np.nan
        feats["avg_cation_oxi"] = np.mean(cation_oxi) if cation_oxi else np.nan
        feats["avg_anion_oxi"] = np.mean(anion_oxi) if anion_oxi else np.nan
        feats["oxi_diff"] = feats["avg_cation_oxi"] - feats["avg_anion_oxi"] if (cation_oxi and anion_oxi) else np.nan
        feats["cation_lonepair_flag"] = cation_lone
        feats["anion_lonepair_flag"] = anion_lone
        rc = self._group_mean(cations, "radius")
        ra = self._group_mean(anions, "radius")
        feats["cation_anion_size_ratio"] = ra / rc if rc and ra and rc > 0 else np.nan

        return [feats[k] for k in self.feature_labels()]

    def feature_labels(self):
        return [
            "delta_en", "delta_radius",
            "delta_valence_s", "delta_valence_p", "delta_valence_d", "delta_valence_f",
            "delta_IE_EA", "delta_polarizability",
            "bond_ionicity",
            "avg_cation_oxi", "avg_anion_oxi", "oxi_diff",
            "cation_lonepair_flag", "anion_lonepair_flag",
            "cation_anion_size_ratio"
        ]


# ===========================
# Main training routine
# ===========================
def train_magpieEX_thermo(data_file="thermalconductivity_K_total3149.json", n_splits=5, length=-1):
    df = pd.read_json(data_file)
    if length > 0:
        df = df.iloc[:length]
    df["structure"] = df["structure"].apply(Structure.from_dict)
    target_col = df.columns[-1]
    print(f"Target column: {target_col}")

    ca_feat = CationAnionContrastFeaturizer()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    maes, r2s = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n=== Fold {fold} ===")
        fold_dir = f"Thermo_MagpieEX_{fold}"
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        def featurize_structure(structure):
            try:
                comp = structure.composition
                magpie_feats = feature_calculators.featurize(comp)
                ca_feats = ca_feat.featurize(str(comp))
                return np.concatenate([magpie_feats, ca_feats])
            except Exception as e:
                print("Featurization failed:", e)
                return np.full(len(feature_calculators.feature_labels()) + len(ca_feat.feature_labels()), np.nan)

        print("Featurizing train set...")
        train_df["features"] = train_df["structure"].progress_apply(featurize_structure)
        print("Featurizing test set...")
        test_df["features"] = test_df["structure"].progress_apply(featurize_structure)

        train_df.to_pickle(os.path.join(fold_dir, "train_features.pkl"))
        test_df.to_pickle(os.path.join(fold_dir, "test_features.pkl"))

        X_train = np.stack(train_df["features"].values)
        y_train = train_df[target_col].values.astype(np.float32)
        X_test = np.stack(test_df["features"].values)
        y_test = test_df[target_col].values.astype(np.float32)

        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        model.fit(X_train.tolist(), y_train.tolist())
        torch.save(model, os.path.join(fold_dir, "tabpfn_model.pt"))

        preds = model.predict(X_test.tolist())
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        maes.append(mae)
        r2s.append(r2)

        pd.DataFrame({
            "target": y_test,
            "prediction": preds
        }).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        with open(os.path.join(fold_dir, "fold_results.txt"), "w") as f:
            f.write(f"Fold {fold}\nMAE={mae:.6f}\nR²={r2:.6f}\n")

        print(f"[Fold {fold}] MAE={mae:.6f}, R²={r2:.6f}")

        del model
        torch.cuda.empty_cache()

    # === Overall summary ===
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)

    print("\n=== Final MagpieEX results on Thermo dataset ===")
    print(f"MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}")

    with open("MagpieEX_results.txt", "w") as f:
        f.write("Per-fold results:\n")
        for i, (m, r) in enumerate(zip(maes, r2s)):
            f.write(f"Fold {i}: MAE={m:.6f}, R²={r:.6f}\n")
        f.write("\n")
        f.write(f"Final MAE mean ± std: {mae_mean:.6f} ± {mae_std:.6f}\n")
        f.write(f"Final R² mean ± std: {r2_mean:.6f} ± {r2_std:.6f}\n")

    print("Saved MagpieEX_results.txt")


if __name__ == "__main__":
    tqdm.pandas()
    train_magpieEX_thermo(data_file="../../farm/thermalconductivity_K_total3149.json", n_splits=5)
