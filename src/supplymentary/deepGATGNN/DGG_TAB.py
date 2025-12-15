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



mb = MatbenchBenchmark(
    autoload=False,
    subset=[
        #"matbench_jdft2d",
        #"matbench_dielectric",
        "matbench_phonons",
        #"matbench_perovskites",
        #"matbench_log_gvrh",
        #"matbench_log_kvrh",
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
                os.chdir(fold_name)
                print("Current working directory:", os.getcwd())

                model_path = f"./tabpfn_model_fold{ii}.pt"
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if False:# os.path.exists(model_path):
                    print(f"Loading pre-trained model for fold {ii}...")
                    my_model = torch.load(model_path)
                else:
                    print(f"Training model for fold {ii}...")
                    my_model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
            
                    # Load training data
                    data_path = "./train_feat.npy"
                    X = np.load(data_path, allow_pickle=True)
                    _, train_outputs = task.get_train_and_val_data(fold)
                    y = train_outputs.to_numpy()
            
                    # Train model
                    my_model.fit(X, y)
            
                    # Save model
                    torch.save(my_model, model_path)
                    print(f"Model saved to {model_path}")
                

                data_path = "./test_feat.npy"
                Xtest = np.load(data_path,allow_pickle=True)
                print(Xtest.shape)  # shape: [N, 5, embedding_dim]
                batch_size = 256  # or smaller if still OOM
                pred_path = f"./predictions_fold{ii}.npy"
                predictions = []


                # Check if predictions already exist
                if os.path.exists(pred_path):
                    # Load existing predictions
                    predictions = np.load(pred_path, allow_pickle=True)
                    start_idx = predictions.shape[0]
                    print(f"Resuming from index {start_idx}, existing predictions shape: {predictions.shape}")
                else:
                    predictions = []
                    start_idx = 0

                num_batches = (Xtest.shape[0] + batch_size - 1) // batch_size
                
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

                all_preds = np.concatenate([predictions], axis=0)
                
                
                #predictions = my_model.predict(Xtest)
                task.record(fold, all_preds)

                del my_model
                gc.collect()
                torch.cuda.empty_cache()
                os.chdir("..")
    # Saving our results
    task_name = task.dataset_name.split("_")[-1]
    mb.to_file(f"DGG_TAB_{task_name}.json")


if __name__ == "__main__":


    train_tasks(mb=mb)

