"""Train ALIGNN model on MatBench dataset."""
# Ref: https://www.nature.com/articles/s41524-021-00650-1

# conda create --name matbench_test python=3.8
# conda activate matbench_test
# pip install alignn matbench dgl-cu111
import shutil
import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matbench.bench import MatbenchBenchmark

#import the crabnet algorithm and its functions
import sys
import os



from tabpfn import TabPFNRegressor
from tabpfn.model.loading import save_tabpfn_model
import gc
import torch
import natsort


mb = MatbenchBenchmark(
    autoload=False,
    subset=[
        #"matbench_jdft2d",
        #"matbench_phonons", 
        #"matbench_dielectric",

        #"matbench_log_gvrh",
        #"matbench_log_kvrh",
        "matbench_perovskites",

        
        # "matbench_mp_e_form",
        # "matbench_mp_gap",
        # "matbench_mp_is_metal",
    ],
)

def train_tasks(mb=None):

    """Train MatBench clalssification and regression tasks."""
    for task in mb.tasks:
        task.load()
        maes = []
        mat_prop = task.dataset_name
        print(f"Training TAB on {mat_prop} dataset")
        #os.chdir("./perovsk_64_1024")
        for ii, fold in enumerate(task.folds):
                fold_name = (
                    task.dataset_name.split("_")[-1]
                    + "_"
                    + str(ii)
                )
                #os.chdir(fold_name)
                print("Current working directory:", os.getcwd())

                model_path = f"./models/{fold_name}/tabpfn_model_fold{ii}.pt"
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if os.path.exists(model_path):
                    print(f"Loading pre-trained model for fold {ii}...")
                    my_model = torch.load(model_path)
                else:
                    print(f"Training model for fold {ii}...")
                    my_model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)


                    # Path to the saved features
                    feat_path = f"./feats/{fold_name}_train_feat.npy"

                    # Load the dictionary
                    data = np.load(feat_path)
 
                    X = data
                    _, train_outputs = task.get_train_and_val_data(fold)
                    y = train_outputs.to_numpy()
            
                    # Train model
                    my_model.fit(X, y)
            
                    # Save model
                    torch.save(my_model, model_path)
                    print(f"Model saved to {model_path}")
                
                # Path to the saved features
                feat_path = f"./feats/{fold_name}_test_feat.npy"    
                # Load the dictionary
                Xtest = np.load(feat_path)
            

                batch_size = 256  # or smaller if still OOM
                pred_path = f"./data/{fold_name}/tab_pred_fold{ii}.npy"
                predictions = []


                #Check if predictions already exist
                if os.path.exists(pred_path):
                    # Load existing predictions
                    predictions = np.load(pred_path, allow_pickle=True)
                    start_idx = predictions.shape[0]
                    print(f"Resuming from index {start_idx}, existing predictions shape: {predictions.shape}")
                else:
                    predictions = []
                    start_idx = 0

                num_batches = (Xtest.shape[0] + batch_size - 1) // batch_size
                
                if start_idx >= Xtest.shape[0]-1:
                    print("All predictions already made. Skipping prediction step.")
                    all_preds = predictions
                else:
                    for i in range(start_idx, Xtest.shape[0], batch_size):
                        batch_num = i // batch_size + 1
                        print(f"Processing batch {batch_num}/{num_batches}...")
                        batch = Xtest[i:i+batch_size]
                        batch_preds = my_model.predict(batch)
                        if isinstance(predictions, list):
                            predictions.append(batch_preds)
                            all_preds = np.concatenate(predictions, axis=0)
                        else:
                            all_preds = np.concatenate([predictions, batch_preds], axis=0)

                        # Save predictions
                        np.save(pred_path, all_preds)
                        print(f"Predictions saved to {pred_path}")
                
                
                #predictions = my_model.predict(Xtest)
                task.record(fold, all_preds)

                del my_model
                gc.collect()
                torch.cuda.empty_cache()
                
    # Saving our results
    task_name = task.dataset_name.split("_")[-1]
    mb.to_file(f"roost_TAB_{task_name}.json")

def train_tasks_feats(mb=None):
    """Train TabPFN on all saved Roost feature types for each MatBench task."""
    feature_types = ["mat", "trunk",  "concat"]  # feature names you saved

    for task in mb.tasks:
        task.load()
        mat_prop = task.dataset_name
        print(f"\n================= Training on {mat_prop} =================")

        for feat_type in feature_types:
            print(f"\n------ Using feature type: {feat_type} ------")
            # create a NEW MatbenchBenchmark each time (prevents duplicate fold errors)

            out_json = f"roost_TAB_{task.dataset_name.split('_')[-1]}_{feat_type}.json"
            # --- check if experiment already done ---
            if os.path.exists(out_json):
                print(f"✅ Skipping {feat_type} — results already exist: {out_json}")
                continue

            print(f"\n------ Using feature type: {feat_type} ------")
            
            mb_local = MatbenchBenchmark(
                autoload=False,
                subset=[
                    "matbench_perovskites",
                    # add others if needed
                ],
            )
            for task in mb_local.tasks:
                task.load()
                mat_prop = task.dataset_name
                print(f"\n🏁 Task: {mat_prop} | Feature type: {feat_type}")

                for ii, fold in enumerate(task.folds):
                    fold_name = f"{task.dataset_name.split('_')[-1]}_{ii}"
                    print(f"\n>>> Fold {ii} | Feature: {feat_type}")
                    print("Current working directory:", os.getcwd())

                    model_dir = f"./models/{fold_name}"
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = f"{model_dir}/tabpfn_model_{feat_type}_fold{ii}.pt"

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # ===== Train or load model =====
                    if os.path.exists(model_path):
                        print(f"Loading pre-trained model for fold {ii} ({feat_type})...")
                        my_model = torch.load(model_path)
                    else:
                        print(f"Training model for fold {ii} ({feat_type})...")
                        my_model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)

                        feat_path = f"./feats/{fold_name}_train_{feat_type}.npy"
                        if not os.path.exists(feat_path):
                            print(f"⚠️ Missing training features: {feat_path}")
                            continue

                        X_train = np.load(feat_path)
                        _, y_train = task.get_train_and_val_data(fold)
                        y_train = y_train.to_numpy()

                        my_model.fit(X_train, y_train)
                        torch.save(my_model, model_path)
                        print(f"✅ Model saved to {model_path}")

                    # ===== Prediction =====
                    feat_path_test = f"./feats/{fold_name}_test_{feat_type}.npy"
                    if not os.path.exists(feat_path_test):
                        print(f"⚠️ Missing test features: {feat_path_test}")
                        continue
                    X_test = np.load(feat_path_test)

                    batch_size = 256
                    pred_dir = f"./data/{fold_name}"
                    os.makedirs(pred_dir, exist_ok=True)
                    pred_path = f"{pred_dir}/tab_pred_{feat_type}_fold{ii}.npy"

                    all_preds = []
                    num_batches = (X_test.shape[0] + batch_size - 1) // batch_size
                    for i in range(0, X_test.shape[0], batch_size):
                        batch_num = i // batch_size + 1
                        print(f"Predicting batch {batch_num}/{num_batches}...")
                        batch = X_test[i:i + batch_size]
                        batch_preds = my_model.predict(batch)
                        all_preds.append(batch_preds)
                        np.save(pred_path, np.concatenate(all_preds, axis=0))
                        print(f"Saved batch {batch_num} predictions to {pred_path}")

                    all_preds = np.concatenate(all_preds, axis=0)
                    task.record(fold, all_preds)

                    del my_model
                    gc.collect()
                    torch.cuda.empty_cache()

            # ===== Save benchmark results for this feature type =====
            task_name = task.dataset_name.split("_")[-1]
            out_json = f"roost_TAB_{task_name}_{feat_type}.json"
            mb_local.to_file(out_json)
            print(f"📁 Saved benchmark results to {out_json}")





if __name__ == "__main__":


    train_tasks_feats(mb=mb)

