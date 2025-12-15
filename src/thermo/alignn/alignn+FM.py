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

def hook(module, input, output,save_path,):
    latent = output.detach().cpu()
    torch.save(latent, save_path)

length = -1
def train_tasks(
    mb=None, config_template="config_example.json", file_format="poscar"
):          
            maes,r2s = [],[]
            for ii in range(5):

                fold_name = f"Thermo_{ii}"  
                os.chdir(fold_name)

                


                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                my_model = TabPFNRegressor(device=device)
                model_path = f"./tabpfn_model.pt"
                data_path = os.path.join("./Ftrain", "features_with_labels.pt")
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

                torch.save(my_model, model_path)
                print(f"Model saved to {model_path}")

                test_path = os.path.join("./Ftest", "features_with_labels.pt")
                test_dict = torch.load(test_path)
                X_test = []
                y_test = []  # optional, only if you want to evaluate

                for d in test_dict.values():
                    latent = d["latent"]
                    target = d["target"]
                    X_test.append(latent.numpy()) 
                    #print(X_test[-1].shape) 
                    y_test.append(target.numpy())  # assuming scalar
                    #print(y_test)
                X_test = np.concatenate(X_test, axis=0).astype(np.float32)
                y_test = np.concatenate(y_test, axis=0).astype(np.float32) 
                #np.array(y_test).astype(np.float32)


                predictions = my_model.predict(X_test)
                pred_path = f"./tab_pred_fold{ii}.npy"
                np.save(pred_path, predictions)
                mae = mean_absolute_error(y_test, predictions)
                maes.append(mae)
                #task.record(fold, pred_vals)
          
                print(
                    "Dataset_name, Fold, MAE=",
                    
                    ii,
                    mae,
                )
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
                
                os.chdir("../")
            maes = np.array(maes)
            print(maes, np.mean(maes), np.std(maes))
            print("+"*40)
            print("+"*40)
    




if __name__ == "__main__":

    train_tasks(mb="0", file_format="poscar")
    