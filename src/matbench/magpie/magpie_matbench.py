
import shutil
import glob
import os
from collections import defaultdict


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, roc_auc_score, r2_score

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY


import torch





from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#torch.use_deterministic_algorithms(False)

mb = MatbenchBenchmark(
    autoload=False,
    subset=[
        # "matbench_jdft2d",
        # "matbench_phonons", 
        # "matbench_dielectric",

        # "matbench_log_gvrh",
        # "matbench_log_kvrh",
        # "matbench_perovskites",

        # "matbench_expt_gap",
        "matbench_steels",
        
        # "matbench_mp_e_form",
        # "matbench_mp_gap",
        # "matbench_mp_is_metal",
    ],
)
import torch
torch.use_deterministic_algorithms(False)
torch.manual_seed(123)
torch.set_default_device("cuda")
torch.set_float32_matmul_precision('high')
from tabpfn import TabPFNRegressor
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf


feature_calculators = MultipleFeaturizer([
    cf.Stoichiometry(),
    cf.ElementProperty.from_preset("magpie"),
    cf.ValenceOrbital(props=['avg']),
    cf.IonProperty(fast=True)])


from pymatgen.core import Element, Composition

# === Custom Featurizer for Cation–Anion Contrast Features ===
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



def train_tasks(
    mb=None, config_template="config_example.json", file_format="poscar"
):
    """Train MatBench clalssification and regression tasks."""
    for task in mb.tasks:
        task.load()
        if task.metadata.task_type == CLF_KEY:
            classification = True
        else:
            classification = False
        # Classification tasks
        if classification:
            # rocs = []
            print("classification task")

        # Regression tasks

        if not classification:
            maes, r2s = [], []
            for ii, fold in enumerate(task.folds):
                #if ii <4:
                #    print("skipping fold", ii)
                #    continue
                train_df= task.get_train_and_val_data(fold, as_type="df")  
                test_df = task.get_test_data(
                    fold, include_target=True, as_type="df"
                )
                
                
                fold_name = (
                    task.dataset_name.split("_")[-1]
                    + "_"
                    + str(ii)
                )

                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)

                target_name = [
                    col
                    for col in train_df.columns
                    if col not in ("id", "structure", "composition")
                ][0]
                
              
                train_feature = []
                train_mp_ids = []   
                train_targets = [] 

                for jj, j in tqdm(train_df.iterrows()):
                    if "structure" in j and j["structure"] is not None:
                        magpie_feats = feature_calculators.featurize(j.structure.composition)
                        # ca_feats = ca_feat.featurize(str(j.structure.composition))
                    elif j.composition:
                        magpie_feats = feature_calculators.featurize(Composition(j.composition))
                        # ca_feats = ca_feat.featurize(Composition(j.composition))
                    # combined_feats = np.concatenate([magpie_feats, ca_feats])
                    # train_feature.append(combined_feats)
                    train_feature.append(magpie_feats)
                    train_mp_ids.append(j.name)
                    train_targets.append(j[target_name])

                test_feature = []
                test_mp_ids = []   
                test_targets = [] 

                for jj, j in tqdm(test_df.iterrows()):
                    if "structure" in j and j["structure"] is not None:
                        magpie_feats = feature_calculators.featurize(j.structure.composition)
                        # ca_feats = ca_feat.featurize(str(j.structure.composition))
                    elif j.composition:
                        magpie_feats = feature_calculators.featurize(Composition(j.composition))
                        # ca_feats = ca_feat.featurize(str(Composition(j.composition)))
                    # combined_feats = np.concatenate([magpie_feats, ca_feats])
                    # test_feature.append(combined_feats)
                    test_feature.append(magpie_feats)
                    test_mp_ids.append(j.name)
                    test_targets.append(j[target_name])



                # --- Train dataframe ---
                train_df_out = pd.DataFrame({
                    "mp_id": train_mp_ids,
                    "target": train_targets,
                    "features": train_feature   # this will be a list/array per row
                })

                # --- Test dataframe ---
                test_df_out = pd.DataFrame({
                    "mp_id": test_mp_ids,
                    "target": test_targets,
                    "features": test_feature
                })

                # Save to CSV (features will be stored as strings, good for inspection)
                train_df_out.to_pickle(f"./{fold_name}/train_features.pkl")
                test_df_out.to_pickle(f"./{fold_name}/test_features.pkl")

                model_path = f"./{fold_name}/tabpfn_model.pt"
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                my_model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)

                
                # Train model
                my_model.fit(train_feature, train_targets)
            
                # Save model
                torch.save(my_model, model_path)
                print(f"Model saved to {model_path}")
                 
              
                predictions = my_model.predict(test_feature)
 
                mae = mean_absolute_error(test_targets, predictions)
                r2 = r2_score(test_targets, predictions)
                maes.append(mae)
                r2s.append(r2)
                 
                task.record(fold, predictions)
          
                print(f"Dataset {task.dataset_name}, fold {fold+1}, MAE={mae}, R2={r2}")


                
                print()
                print()
                print()
                print()
                print()
                print()
                print("+"*40)
                print("+"*40)

                #break



            # maes = np.array(maes)
            print(f"Dataset {task.dataset_name} Final MAE across folds:", np.mean(maes), "±", np.std(maes))
            print(f"Dataset {task.dataset_name} final R2 across folds:", np.mean(r2s), "±", np.std(r2s))


            print("+"*40)
            print("+"*40)

            task_name = task.dataset_name.split("_")[-1]
            mb.to_file(f"tabpfn_compo_extra_{task_name}.json")

if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    #config = loadjson(config_template)
    train_tasks(mb=mb, file_format="poscar")

    
