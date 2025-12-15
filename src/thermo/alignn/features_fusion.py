
import shutil
import glob
import os
from collections import defaultdict
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或 ":16:8"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import numpy as np
import pandas as pd

from jarvis.core.atoms import pmg_to_atoms
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph


from jarvis.db.jsonutils import dumpjson, loadjson
from sklearn.metrics import mean_absolute_error, roc_auc_score

sys.path.append(os.path.abspath("../../phase100_alignn_eval/alignn"))
from alignn.models.alignn_atomwise   import ALIGNNAtomWise, ALIGNNAtomWiseConfig


from tabpfn import TabPFNRegressor

import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

from contextlib import AbstractContextManager
from typing import List, Tuple, Dict, Any
import torch
from torch import nn
from tqdm import tqdm

import torch
#torch.use_deterministic_algorithms(True)
torch.manual_seed(666)

def clean():
    folders_to_delete = [
                        "sampletest_data",
                        "sampletrain_data",
                        "sampleval_data"
    ]
    for folder in folders_to_delete:
        if os.path.exists(folder):
                            shutil.rmtree(folder)
                            print(f"Deleted folder: {folder}")
        else:
                            print(f"Folder does not exist: {folder}")
def hook(module, input, output,save_path,):
    latent = output.detach().cpu()
    torch.save(latent, save_path)


# --- activation collector for ALIGNNAtomWise  ---

class ALIGNNAtomWiseActivationCollector(AbstractContextManager):
    """
    Collects mean activations (dim=0) for x/y/z across:
      - atom_embedding  -> x0
      - edge_embedding  -> y0
      - angle_embedding -> z0
      - each ALIGNNConv layer -> (x_i, y_i, z_i)
      - each GCN EdgeGatedGraphConv layer -> (xg_i, yg_i)
    Produces act_list_x, act_list_y, act_list_z (list of 2D tensors [1, hidden_dim])
    without modifying ALIGNNAtomWise code.
    """
    def __init__(self, model, keep_device: bool = False, detach: bool = True):
        self.model = model
        self.keep_device = keep_device
        self.detach = detach
        self.handles: List[Any] = []

        # public buffers
        self.act_list_x: List[torch.Tensor] = []
        self.act_list_y: List[torch.Tensor] = []
        self.act_list_z: List[torch.Tensor] = []

        # register hooks
        self._register_hooks()

    def _maybe_process(self, t: torch.Tensor) -> torch.Tensor:
        if self.detach:
            t = t.detach()
        t = t.mean(dim=0, keepdim=True)  # [1, hidden]
        if not self.keep_device:
            t = t.cpu()
        return t

    def _register_hooks(self):
        m = self.model

        # 1) embeddings: x0, y0, z0
        if hasattr(m, "atom_embedding"):
            self.handles.append(
                m.atom_embedding.register_forward_hook(
                    lambda mod, inp, out: self.act_list_x.append(self._maybe_process(out))
                )
            )
        if hasattr(m, "edge_embedding"):
            self.handles.append(
                m.edge_embedding.register_forward_hook(
                    lambda mod, inp, out: self.act_list_y.append(self._maybe_process(out))
                )
            )
        if hasattr(m, "angle_embedding"):
            self.handles.append(
                m.angle_embedding.register_forward_hook(
                    lambda mod, inp, out: self.act_list_z.append(self._maybe_process(out))
                )
            )

        # 2) ALIGNNConv layers: output is (x, y, z)
        if hasattr(m, "alignn_layers"):
            for i, layer in enumerate(m.alignn_layers):
                def _make_alignn_hook(idx):
                    def _hook(mod, inp, out):
                        # out is a tuple: (x, y, z)
                        x, y, z = out
                        self.act_list_x.append(self._maybe_process(x))
                        self.act_list_y.append(self._maybe_process(y))
                        self.act_list_z.append(self._maybe_process(z))
                    return _hook
                self.handles.append(layer.register_forward_hook(_make_alignn_hook(i)))

        # 3) GCN EdgeGatedGraphConv layers (node/edge updates): output is (x, y)
        if hasattr(m, "gcn_layers"):
            for i, layer in enumerate(m.gcn_layers):
                def _make_gcn_hook(idx):
                    def _hook(mod, inp, out):
                        # out is a tuple: (x, y)
                        x, y = out
                        self.act_list_x.append(self._maybe_process(x))
                        self.act_list_y.append(self._maybe_process(y))
                    return _hook
                self.handles.append(layer.register_forward_hook(_make_gcn_hook(i)))

    def reset(self):
        self.act_list_x.clear()
        self.act_list_y.clear()
        self.act_list_z.clear()

    def get_lists(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        return self.act_list_x, self.act_list_y, self.act_list_z

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don't suppress exceptions



###process features

def _strip_ext(name: str) -> str:
    stem, ext = os.path.splitext(name)
    return stem if ext else name  # allow digit-only filenames

def _load_labels(split_dir: str, strip_vasp: bool = True) -> dict:
    """Read {split_dir}/id_prop.csv -> {id: target}"""
    csv_path = os.path.join(split_dir, "id_prop.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path, header=None).rename(columns={0: "id", 1: "prop"})
    if strip_vasp:
        df["id"] = df["id"].map(lambda x: str(x).rstrip(".vasp"))
    return {str(r["id"]): float(r["prop"]) for _, r in df.iterrows()}

def _pick_row(df: pd.DataFrame, row_idx_1based: int) -> np.ndarray:
    """Get 1-based row; if out of range use last row."""
    if len(df) == 0:
        raise ValueError("Empty feature CSV.")
    idx0 = min(max(row_idx_1based - 1, 0), len(df) - 1)
    # First 256 columns assumed to be '0'..'255'
    cols = [str(i) for i in range(256) if str(i) in df.columns]
    if len(cols) < 1:
        # fallback: take first 256 columns whatever they are
        cols = df.columns[:256].tolist()
    return df.loc[idx0, cols].to_numpy(dtype=np.float32)

def _collect_split_features(fold_name: str, split: str, ck_x=9, ck_y=9, ck_z=5) -> dict:
    """
    From {fold}/{split}/id_prop.csv and {fold}/TF{split}/*_{x,y,z}.csv,
    build dict id -> {"latent": FloatTensor[768], "target": float or None}
    """
    split_dir = os.path.join(fold_name, split)
    tf_dir = os.path.join(fold_name, f"TF{split}")
    labels = _load_labels(split_dir)

    data = {}
    if not os.path.isdir(tf_dir):
        return data

    for fname in os.listdir(tf_dir):
        if not (fname.endswith("_x.csv") or fname.endswith("_y.csv") or fname.endswith("_z.csv")):
            continue
        base = fname.rsplit("_", 1)[0]  # strip _x/_y/_z
        # process each base only once
        if base in data:
            continue

        x_path = os.path.join(tf_dir, f"{base}_x.csv")
        y_path = os.path.join(tf_dir, f"{base}_y.csv")
        z_path = os.path.join(tf_dir, f"{base}_z.csv")
        if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(z_path)):
            # require all three to form 768-dim vector
            continue

        dfx = pd.read_csv(x_path)
        dfy = pd.read_csv(y_path)
        dfz = pd.read_csv(z_path)

        vx = _pick_row(dfx, ck_x)  # 256
        vy = _pick_row(dfy, ck_y)  # 256
        vz = _pick_row(dfz, ck_z)  # 256
        feat = np.concatenate([vx, vy, vz], axis=0).astype(np.float32)  # 768

        key = _strip_ext(base)  # to match id_prop.csv keys
        target = labels.get(key, None)

        data[key] = {
            "latent": torch.tensor(feat, dtype=torch.float32),
            "target": None if target is None else float(target),
        }
    return data



def train_tasks(
    mb=None, config_template="./config_example.json", file_format="poscar"
):
            maes = []
            for ii in range(5):
                fold_name = f"Thermo_{ii}"   

                # -------------------------
                # 1) Load ALIGNNAtomWise
                # -------------------------


                device = "cpu"
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                
                
                filename = fold_name+"/training/best_model.pt"
                rest_config = loadjson(filename.replace("best_model.pt", "config.json"))

                tmp = ALIGNNAtomWiseConfig(**rest_config["model"])
                model = ALIGNNAtomWise(tmp)  # config.model)
                model.load_state_dict(torch.load(filename, map_location=device))
                model = model.to(device)
                model.eval()

                # -------------------------
                # 2) Extract features
                # -------------------------

                train_dir = os.path.join(fold_name, "train")
                output_path = fold_name + "/TFtrain"
                os.makedirs(output_path, exist_ok=True)
                for file_name in tqdm(os.listdir(train_dir), desc="Processing train files"):
                    # skip the CSV file
                    if file_name == "id_prop.csv":
                        continue  


                    base_name = os.path.splitext(file_name)[0]
                    # --- skip if already exists ---
                    already_done = all(
                        os.path.exists(f"{output_path}/{base_name}_{sfx}.csv")
                        for sfx in ["x", "y", "z"]
                    )
                    if already_done:
                        print(f"Skip {base_name}, features already exist.")
                        continue
                    
                    
                    file_path = os.path.join(train_dir, file_name)
                    atoms = Atoms.from_poscar(file_path)
                    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(8))
                    lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32, device=device)
                
                    collector = ALIGNNAtomWiseActivationCollector(model, keep_device=False, detach=True)
                    with torch.no_grad():
                    
                        _ = model([g.to(device), lg.to(device), lat])  
               

                
                    act_list_x, act_list_y, act_list_z = collector.get_lists()
                    collector.close()
                    base_name = os.path.splitext(file_name)[0]


                    # --- save activation traces
                
                    def save_act_list(act_list, suffix):
                        arrs = [t.detach().cpu().numpy() for t in act_list]
                        if arrs:  # avoid empty list
                            np_act = np.concatenate(arrs, axis=0)
                            pd.DataFrame(np_act).to_csv(f"{output_path}/{base_name}_{suffix}.csv", index=False)

                    save_act_list(act_list_x, "x")
                    save_act_list(act_list_y, "y")
                    save_act_list(act_list_z, "z")

                test_dir = os.path.join(fold_name, "test")
                output_path = fold_name + "/TFtest"
                os.makedirs(output_path, exist_ok=True)

                for file_name in tqdm(os.listdir(test_dir), desc="Processing test files"):

                    # skip the CSV file
                    if file_name == "id_prop.csv":
                        continue  


                    base_name = os.path.splitext(file_name)[0]
                    # --- skip if already exists ---
                    already_done = all(
                        os.path.exists(f"{output_path}/{base_name}_{sfx}.csv")
                        for sfx in ["x", "y", "z"]
                    )
                    if already_done:
                        print(f"Skip {base_name}, features already exist.")
                        continue


                    file_path = os.path.join(test_dir, file_name)
                    atoms = Atoms.from_poscar(file_path)
                    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(8))
                    lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32, device=device)
                
                    collector = ALIGNNAtomWiseActivationCollector(model, keep_device=False, detach=True)
                    with torch.no_grad():
                    
                        _ = model([g.to(device), lg.to(device), lat])  
               

                
                    act_list_x, act_list_y, act_list_z = collector.get_lists()
                    collector.close()
                    base_name = os.path.splitext(file_name)[0]


                    # --- save activation traces

                    def save_act_list(act_list, suffix):
                        arrs = [t.detach().cpu().numpy() for t in act_list]
                        if arrs:  # avoid empty list
                            np_act = np.concatenate(arrs, axis=0)
                            pd.DataFrame(np_act).to_csv(f"{output_path}/{base_name}_{suffix}.csv", index=False)

                    save_act_list(act_list_x, "x")
                    save_act_list(act_list_y, "y")
                    save_act_list(act_list_z, "z")
                

                #XYZ features saved for train and test sets 

                # -------------------------
                # 3) Process features
                # -------------------------

                # Assemble and save per-fold train/test feature tensors
                ftrain_dir = os.path.join(fold_name, "Ftrain")
                ftest_dir  = os.path.join(fold_name, "Ftest")
                os.makedirs(ftrain_dir, exist_ok=True)
                os.makedirs(ftest_dir, exist_ok=True)

                train_out = os.path.join(fold_name, "TF_train_features_with_labels.pt")
                test_out  = os.path.join(fold_name, "TF_test_features_with_labels.pt")

                if not os.path.exists(train_out):
                    train_dict = _collect_split_features(fold_name, split="train", ck_x=9, ck_y=9, ck_z=5)
                    torch.save(train_dict, train_out)
                    print(f"Saved {train_out}")
                else:
                    print(f"Skip saving, already exists: {train_out}")
                    train_dict = torch.load(train_out)


                if not os.path.exists(test_out):
                    test_dict  = _collect_split_features(fold_name, split="test", ck_x=9, ck_y=9, ck_z=5)
                    torch.save(test_dict, test_out)
                    print(f"Saved {test_out}")
                else:
                    print(f"Skip saving, already exists: {test_out}")
                    test_dict = torch.load(test_out)

                print(f"[{fold_name}] Saved TabPFN features: "
                    f"{len(train_dict)} train, {len(test_dict)} test entries.")

                #feature processed Next TabPFN

                # -------------------------
                # 3) Train TabPFN
                # -------------------------

                print("+"*80)
                print()
                print()
                 # Convert dict → (X, y)
                def dict_to_xy(d: dict):
                    X_rows, y_vals = [], []
                    for k, v in d.items():
                        if v["target"] is None:
                            # skip unlabeled entries (shouldn't happen if id_prop.csv exists)
                            continue
                        X_rows.append(v["latent"])
                        y_vals.append(v["target"])
                    if not X_rows:
                        raise RuntimeError("No labeled entries found. Check id_prop.csv and IDs.")
                    X = np.stack(X_rows, axis=0).astype(np.float32)
                    y = np.array(y_vals, dtype=np.float32)
                    return X, y
                
                X_train, y_train = dict_to_xy(train_dict)
                X_test,  y_test  = dict_to_xy(test_dict)
                print(f"[Fold {ii}] Train X shape: {X_train.shape}, Test X shape: {X_test.shape}")


                model_path = os.path.join(fold_name, f"TF_tabpfn_model_{ii}.pt")

                tab_device = device  # use the same device
                model_pfn = TabPFNRegressor(device=tab_device,ignore_pretraining_limits=True)
                if not os.path.exists(model_path):
                    model_pfn.fit(X_train, y_train)
                    torch.save(model_pfn, model_path)
                    print(f"Saved new model to {model_path}")
                else:
                    # load the saved model
                    model_pfn = torch.load(model_path, map_location=device)
                    print(f"Skip training, model already exists at {model_path}")

                
                preds = model_pfn.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                maes.append(mae)

                df_preds = pd.DataFrame({"prediction": preds})
                csv_path = os.path.join(fold_name, f"TF_tab_pred_fold{ii}.csv")
                df_preds.to_csv(csv_path, index=False)
                print(f"Predictions saved to {csv_path}")

                print(f"[Fold {ii}] MAE: {mae:.6f}")



                del model_pfn
                torch.cuda.empty_cache()   # if it was on GPU

  




                

         
                



            mae_path = "mae_scores.pt"
            torch.save(maes, mae_path)
            maes = np.array(maes)
            print(maes, np.mean(maes), np.std(maes))
            print(f"Saved MAE scores dict to {mae_path}")
            print("+"*40)
            print("+"*40)

            
            
if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    #config = loadjson(config_template)
    train_tasks(mb="mb", file_format="poscar")

    