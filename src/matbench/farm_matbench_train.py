import os
import numpy as np
import pandas as pd
import dgl
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from pymatgen.io.vasp import Poscar
from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import pmg_to_atoms

from matbench.bench import MatbenchBenchmark

from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.dataset import get_torch_dataset

from tabpfn import TabPFNRegressor


# -------------------------
# Utilities
# -------------------------

# def extract_features(model, loader, device="cuda"):
#     """Extract intermediate embeddings from ALIGNN model."""
#     model.eval()
#     feats, targets = [], []

#     embeddings = {}

#     # Hook into final readout layer (or fc fallback)
#     def hook(module, input, output):
#         if isinstance(output, tuple):        # many ALIGNN layers return (x, y)
#             x, _ = output
#             embeddings['feat'] = x.detach().cpu()  # keep atom/node features
#         else:
#             embeddings['feat'] = output.detach().cpu()

    ## try read_out layer
    # if hasattr(model, "readout"):
    #     handle = model.readout.register_forward_hook(hook)
    # elif hasattr(model, "fc"):
    #     handle = model.fc.register_forward_hook(hook)
    # else:
    #     raise RuntimeError("Model has no 'readout' or 'fc' layer to hook.")

    
    # with torch.no_grad():
    #     for (g, latt), y in loader:
    #         g, latt, y = g.to(device), latt.to(device), y.to(device)
    #         _ = model((g, latt))
    #         pooled = embeddings['feat']
    #         feats.append(pooled.numpy())
    #         targets.append(y.cpu().numpy())

    # handle.remove()
    # return np.vstack(feats), np.concatenate(targets)

    # # try gcn_layers
    # handle = model.gcn_layers[-1].register_forward_hook(hook)
    
    # with torch.no_grad():
    #     for (g, latt), y in loader:
    #         g, latt, y = g.to(device), latt.to(device), y.to(device)
    #         _ = model((g, latt))

    #         node_feats = embeddings['feat']  # shape: [n_nodes_in_batch, hidden_dim]

    #         # split node feats back into graphs
    #         bg = g.to("cpu")  # batched graph
    #         graph_feats = []
    #         for nid in range(bg.batch_size):
    #             node_idx = (bg.batch_num_nodes()[:nid].sum().item(),
    #                         bg.batch_num_nodes()[:nid+1].sum().item())
    #             nodes = node_feats[node_idx[0]:node_idx[1]]
    #             pooled = nodes.mean(dim=0)  # mean pooling
    #             graph_feats.append(pooled.numpy())

    #         feats.append(np.vstack(graph_feats))
    #         targets.append(y.cpu().numpy())

    # handle.remove()
    # return np.vstack(feats), np.concatenate(targets)

    # # # try alignn_layers 
    # # handle = model.alignn_layers[-1].register_forward_hook(hook)


def extract_features(model, loader, device="cuda", layer="readout", pool="mean"):
    """
    Extract graph-level embeddings from ALIGNN.

    Args:
        model: Trained ALIGNN model
        loader: DataLoader
        device: "cuda" or "cpu"
        layer: str, one of {"readout", "gcn_last", "alignn_last"}
        pool: str, how to pool node embeddings if using gcn/alignn layers {"mean", "sum", "max"}
    """
    model.eval()
    feats, targets = [], []
    embeddings = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            if len(output) == 2:         # GCN layers
                x, y = output
            elif len(output) == 3:       # ALIGNN layers
                x, y, z = output
            else:
                raise ValueError(f"Unexpected output tuple length {len(output)}")
            embeddings['feat'] = x.detach().cpu()  # keep node features
        else:
            embeddings['feat'] = output.detach().cpu()

    # attach hook

    if layer == "readout": ##40.88
        handle = model.readout.register_forward_hook(hook)
    elif layer == "gcn_last": ##40.88
        handle = model.gcn_layers[-1].register_forward_hook(hook)
    elif layer == "alignn_last": ##40.98
        handle = model.alignn_layers[-1].register_forward_hook(hook)
    else:
        raise ValueError(f"Unsupported layer choice: {layer}")

    with torch.no_grad():
        for (g, latt), y in loader:
            g, latt, y = g.to(device), latt.to(device), y.to(device)
            _ = model((g, latt))
            feat = embeddings['feat']

            # If node-level features, pool into graph-level
            if feat.ndim == 2 and feat.shape[0] != y.shape[0]:
                bg = g.to(device)
                graph_feats = []
                node_counts = bg.batch_num_nodes().tolist()
                start = 0
                for count in node_counts:
                    nodes = feat[start:start+count]
                    if pool == "mean":
                        pooled = nodes.mean(dim=0)
                    elif pool == "sum":
                        pooled = nodes.sum(dim=0)
                    elif pool == "max":
                        pooled, _ = nodes.max(dim=0)
                    elif pool == "mean+max":
                        pooled = torch.cat([nodes.mean(dim=0), nodes.max(dim=0)[0]])
                    else:
                        raise ValueError(f"Unsupported pool: {pool}")
                    graph_feats.append(pooled.numpy())
                    start += count
                feat = np.vstack(graph_feats)

            feats.append(feat if isinstance(feat, np.ndarray) else feat.numpy())
            targets.append(y.cpu().numpy())

    handle.remove()
    return np.vstack(feats), np.concatenate(targets)



def collate_fn(batch):
    graphs, lattices, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    lattices = torch.stack(lattices)
    labels = torch.stack(labels)
    return (batched_graph, lattices), labels


# -------------------------
# Main function
# -------------------------

def run_tabpfn_with_matbench(
    dataset="matbench_jdft2d",
    device="cuda",
    property_name="exfoliation_en",
):
    mb = MatbenchBenchmark(subset=[dataset])

    maes, r2s = [], []

    for task in mb.tasks:
        task.load()
        target = [
            c for c in task.metadata.columns if c not in ("id", "structure", "composition")
        ][0]

        for ii, fold in enumerate(task.folds):
            print(f"\n=== {dataset}, Fold {ii} ===")

            # official Matbench train/test splits
            train_df = task.get_train_and_val_data(fold, as_type="df")
            test_df = task.get_test_data(fold, include_target=True, as_type="df")

            # load ALIGNN pretrained model for this fold
            fold_dir = f"{dataset}_{property_name}_outdir_{ii}"
            config_path = os.path.join(fold_dir, "config.json")
            model_path = os.path.join(fold_dir, "best_model.pt")

            config = loadjson(config_path)
            model_cfg = ALIGNNAtomWiseConfig(**config["model"])
            model = ALIGNNAtomWise(model_cfg)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            # helper to convert dataframe to ALIGNN dataset
            def df_to_dataset(df, split_name):
                dataset_list = []
                for idx, row in df.iterrows():
                    atoms = pmg_to_atoms(row.structure)
                    dataset_list.append({
                        "jid": str(idx),
                        "atoms": atoms.to_dict(),
                        "target": row[target]
                    })
                return get_torch_dataset(
                    dataset=dataset_list,
                    target="target",
                    neighbor_strategy="k-nearest",
                    atom_features="cgcnn",
                    output_dir=fold_dir,
                    tmp_name=f"{split_name}_fold{ii}",
                )

            train_dataset = df_to_dataset(train_df, "train")
            test_dataset = df_to_dataset(test_df, "test")

            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
            )
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
            )

            # extract ALIGNN embeddings
            X_train, y_train = extract_features(model, train_loader, device, layer="readout")
            X_test, y_test = extract_features(model, test_loader, device, layer="readout")

            tbf = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
            tbf.fit(X_train, y_train)
            
            torch.save(tbf,  f"{fold_dir}/farm_model.pt")
            print(f"Model saved to {fold_dir}/farm_model.pt")
            y_pred = tbf.predict(X_test)

            # evaluate
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            maes.append(mae)
            r2s.append(r2)

            print(f"Fold {ii}: MAE={mae:.4f}, R2={r2:.4f}")

    print("\n=== Final Results ===")
    print("MAE across folds:", np.mean(maes), "±", np.std(maes))
    print("R² across folds:", np.mean(r2s), "±", np.std(r2s))


if __name__ == "__main__":
    # Example usage:
    task_id = 5
           
    subsets=[
        "matbench_jdft2d",
        "matbench_dielectric",
        "matbench_phonons",
        "matbench_perovskites",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
        # "matbench_mp_e_form",
        # "matbench_mp_gap",
        # "matbench_mp_is_metal",
    ]
    targets=[
        "exfoliation_en",
        "n",
        "last_phdos_peak",
        "e_form", 
        "log10-G_VRH-",
        "log10-K_VRH-"  
    ]

    run_tabpfn_with_matbench(
        dataset=subsets[task_id],
        device="cuda:3",
        property_name=targets[task_id],
    )