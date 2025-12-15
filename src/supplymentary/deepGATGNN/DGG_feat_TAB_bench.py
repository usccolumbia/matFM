"""Train ALIGNN model on MatBench dataset."""
# Ref: https://www.nature.com/articles/s41524-021-00650-1

# conda create --name matbench_test python=3.8
# conda activate matbench_test
# pip install alignn matbench dgl-cu111
import shutil
import glob
import os
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson, loadjson
from sklearn.metrics import mean_absolute_error, roc_auc_score

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY

from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.data import get_train_val_loaders

from tabpfn import TabPFNRegressor
from tabpfn.model.loading import save_tabpfn_model
import torch

mb = MatbenchBenchmark(
    autoload=False,
    subset=[
        "matbench_jdft2d",
        # "matbench_dielectric",
        #"matbench_phonons",
        # "matbench_perovskites",
        # "matbench_log_gvrh",
        # "matbench_log_kvrh",
        # "matbench_mp_e_form",
        # "matbench_mp_gap",
        # "matbench_mp_is_metal",
    ],
)



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
        # TODO: shorten the script by taking out repetitive lines
        if not classification:
            maes = []
            for ii, fold in enumerate(task.folds):
                train_df = task.get_train_and_val_data(fold, as_type="df")
                test_df = task.get_test_data(
                    fold, include_target=True, as_type="df"
                )
                # Name of the target property
                target = [
                    col
                    for col in train_df.columns
                    if col not in ("id", "structure", "composition")
                ][0]
                # Making sure there are spaces or parenthesis which
                # can cause issue while creating folder
                fold_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_fold_"
                    + str(ii)
                )
                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)
                os.chdir(fold_name)
                # ALIGNN requires the id_prop.csv file
                f = open("id_prop.csv", "w")
                for jj, j in train_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                # There is no pre-defined validation splt, so we will use
                # a portion of training set as validation set, and
                # keep test set intact
                val_df = train_df[0 : len(test_df)]
                for jj, j in val_df.iterrows():
                    # for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                for jj, j in test_df.iterrows():
                    id = j.name
                    atoms = pmg_to_atoms(j.structure)
                    pos_name = id
                    atoms.write_poscar(pos_name)
                    val = j[target]
                    line = str(pos_name) + "," + str(val) + "\n"
                    f.write(line)
                n_train = len(train_df)
                n_val = len(val_df)
                n_test = len(test_df)
                config = loadjson(config_template)
                config["n_train"] = n_train
                config["n_val"] = n_val
                config["n_test"] = n_test
                config["keep_data_order"] = True
                config["batch_size"] = 32
                config["epochs"] = 500
                fname = "config_fold_" + str(ii) + ".json"
                dumpjson(data=config, filename=fname)
                f.close()
                os.chdir("..")
                outdir_name = (
                    task.dataset_name
                    + "_"
                    + target.replace(" ", "_")
                    .replace("(", "-")
                    .replace(")", "-")
                    + "_outdir_"
                    + str(ii)
                )
                cmd = (
                    "python ./alignn/alignn/feat_alignn.py --root_dir "
                    + fold_name
                    + " --config "
                    + fold_name
                    + "/"
                    + fname
                    + " --file_format="
                    + file_format
    
                    + " --restart_model_path="
                    + outdir_name+ "/best_model.pt"                    
                    + " --output_dir="
                    + outdir_name
                )

                data_path = os.path.join(outdir_name, "training_features_with_labels.pt")

                if os.path.exists(data_path):
                    print(f"Feature file already exists at {data_path}, skipping feature extraction.")
                else:
                    print(cmd)
                    os.system(cmd)
                    print("==" * 50)
                    print("alignn model feature extraction done for fold", ii)
                print("==" * 50)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                my_model = TabPFNRegressor(device=device)

                data_path = outdir_name + "/training_features_with_labels.pt"
                data_dict = torch.load(data_path)  # 每个键是 jid，值是 {latent, target}
                
                X = []
                y = []

                for d in data_dict.values():
                    latent_batch = d["latent"]    # shape: [32, 256]
                    target_batch = d["target"]    # shape: [32]

                    for latent_vec, target_val in zip(latent_batch, target_batch):
                        X.append(latent_vec.cpu().numpy())     # single [256] vector
                        y.append(target_val.item())            # single float

                X = np.stack(X).astype(np.float32)
                y = np.array(y).astype(np.float32)
                print("X shape:", X.shape)
                my_model.fit(X,y)



                test_path = outdir_name + "/testing_features_with_labels.pt"
                test_dict = torch.load(test_path)
                X_test = []
                y_test = []  # optional, only if you want to evaluate

                for d in test_dict.values():
                    latent = d["latent"]
                    target = d["target"]
                    X_test.append(latent.view(-1).numpy())
                    y_test.append(target.view(-1).numpy()[0])  # assuming scalar
                X_test = np.stack(X_test).astype(np.float32)
                y_test = np.array(y_test).astype(np.float32)


                predictions = my_model.predict(X_test)

                # List of folder paths to delete
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
                # Record your data!
                task.record(fold, predictions)



if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    config = loadjson(config_template)
    train_tasks(mb=mb, config_template=config_template, file_format="poscar")

    run_dir = "./"
    # run_dir = "/wrk/knc6/matbench/benchmarks/matbench_v0.1_alignn"
    mb.add_metadata({"algorithm": "ALIGNN_feat_tabpfn"})
    mb.to_file("feat_extract.json.gz")
    print(mb.scores)