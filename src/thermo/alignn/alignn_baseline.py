
import shutil
import glob
import os
from collections import defaultdict


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import numpy as np
import pandas as pd
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson, loadjson
from sklearn.metrics import mean_absolute_error, roc_auc_score

import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")
from pymatgen.core import Structure
from sklearn.model_selection import KFold
n_splits = 5

import torch
torch.use_deterministic_algorithms(True)
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




def train_tasks(
    mb=None, config_template="./config_example.json", file_format="poscar", length=-1
):
   
    data_file = "thermalconductivity_K_total3149.json" 

    df = pd.read_json(data_file)
    # df = df_full[:1000]  # subset for testing
    df["structure"] = df["structure"].apply(Structure.from_dict)
    #df = df.reset_index(drop=True)

    target_col = df.columns[-1]
    # print(target_col)
    # exit()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    maes,r2s = [],[]
    results = defaultdict()


    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):

      
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        print()
        fold_name = f"Thermo_{fold}"  
        if not os.path.exists(fold_name):
                    os.makedirs(fold_name)
                
                
        def write_data(datdf, folder_path):
                os.makedirs(folder_path, exist_ok=True)
                targets_path = os.path.join(folder_path, "id_prop.csv")
                    
                with open(targets_path, "w") as f:
                        for jj, j in datdf.iterrows():
                            id = j.name
                            atoms = pmg_to_atoms(j.structure)
                            pos_name = os.path.join(folder_path, str(id))#+ ".poscar"
                            atoms.write_poscar(pos_name)

                            val = j[target_col]
                            line = f"{id},{val}\n"
                            f.write(line)
                # Write files

        train_path = os.path.join(fold_name, "train")
        test_path = os.path.join(fold_name, "test")

        if not os.path.exists(train_path+"/id_prop.csv"):
            write_data(train_df, train_path)
        else:
            print(f"Skipping train data, found {train_path}")
        if not os.path.exists(test_path+"/id_prop.csv"):
            write_data(test_df, test_path)
        else:
            print(f"Skipping test data, found {test_path}")
                
                #os.chdir(fold_name)

        n_train = len(train_df)
        #n_val = len(val_df)
        n_test = len(test_df)
        config = loadjson(config_template)
        config["n_train"] = int(n_train*.9)
        config["n_val"] = int(n_train*.05)
        config["n_test"] = int(n_train*.05)
        config["filename"] = fold_name+"_training_" 
        config["keep_data_order"] = True
        config["batch_size"] = 256
        config["epochs"] = 500
        tr_conf = "config_" + str(fold) +"_train" ".json"
        tr_conf_dir = fold_name+"/"+tr_conf
        dumpjson(data=config, filename=tr_conf_dir)


        config = loadjson(config_template)
        config["n_train"] = 0
        config["n_val"] = 0
        config["n_test"] = n_test
        config["keep_data_order"] = True
        config["batch_size"] = n_test
        config["epochs"] = 0
        config["filename"] = fold_name+"_testing_" 
        te_conf = "config_" + str(fold) +"_test" ".json"
        te_conf_dir = fold_name+"/"+te_conf
        dumpjson(data=config, filename=te_conf_dir)

        config = loadjson(config_template)
        config["n_train"] = n_train
        config["n_val"] = None
        config["n_test"] = None
        config["keep_data_order"] = True
        config["batch_size"] = 128
        config["filename"] = fold_name+"_feat_train_" 
        fer_conf = "config_" + str(fold) + "_feat_train.json"
        fer_conf_dir = fold_name+"/"+fer_conf

        dumpjson(data=config, filename=fer_conf_dir)

        config = loadjson(config_template)
        config["n_train"] = n_test
        config["n_val"] = None
        config["n_test"] = None
        config["keep_data_order"] = True
        config["batch_size"] = 128
        config["filename"] = fold_name+"_feat_test_" 
        fte_conf = "config_" + str(fold) + "_feat_test.json"
        fte_conf_dir = fold_name+"/"+fte_conf

        dumpjson(data=config, filename=fte_conf_dir)

        os.chdir(fold_name)
        print("Current working directory:", os.getcwd())
        
        alignn_dir = "../../phase100_alignn_eval/alignn/"
        #alignn_dir = "../../alifeat/alignn/"
        #envir_setting  = "PYTHONPATH=/home/qinyang/work/alifeat:$PYTHONPATH \
        #                    CUDA_VISIBLE_DEVICES=1"
        
        ####### training command ########
        print("starting training for fold", fold)
        traincmd = (
                #envir_setting+
            " python "+ alignn_dir+"train_alignn.py --root_dir "
            + "./"+"train"
            + " --config "
            + "./"+tr_conf
            + " --file_format="
            + file_format
            + " --keep_data_order=True"
            + " --output_dir="
            + "./"+"training"
        )
        print(traincmd)
        best_model_path =  "./training/best_model.pt"
        print("Best model path:", best_model_path)
        if os.path.exists(best_model_path):
                    print(f"Found {best_model_path}, skipping training.")
        else:
                    print("Training model...")
                    os.system(traincmd)
                #print(traincmd)
                #subprocess.run(traincmd, shell=True)# this is quiet
                

        print("+"*80)
        print()
        print()
        
                ###### testing command #########
        os.makedirs("testing", exist_ok=True)
        testcmd = (
                    "python "+ alignn_dir+"test_alignn.py --root_dir "
                    + "./test"
                    + " --config "
                    + "./"+te_conf
                    + " --file_format="
                    + file_format
                    + " --restart_model_path="
                    + best_model_path
                    + " --keep_data_order=True"
                    + " --output_dir="
                    + "./testing"
                )
                #print(testcmd)
        os.system(testcmd)

                
        print("fold=",fold)
        print("+"*80)
        print()
        print()
        
                ###### feature extract command #########


        train_feat_cmd = (
                        "python "+ alignn_dir+"feat_alignn.py --root_dir "
                    + "./train"
                    + " --config "
                    + "./"+fer_conf
                    + " --file_format="
                    + file_format
    
                    + " --restart_model_path="
                    + best_model_path                    
                    + " --output_dir="
                    + "./Ftrain"
                )
                
        test_feat_cmd = (
                        "python "+ alignn_dir+"feat_alignn.py --root_dir "
                    + "./test"
                    + " --config="
                    + "./"+fte_conf
                    + " --file_format="
                    + file_format
                    + " --restart_model_path="
                    +  best_model_path                    
                    + " --output_dir="
                    + "./Ftest"
                )


                #print(train_feat_cmd)
        os.system(train_feat_cmd)
                
                #print(test_feat_cmd)
        os.system(test_feat_cmd)
                


        print("+"*80)
        print()
        print()
     


        test_csv =  "./testing/prediction_full_test_set.csv"
         
        outdf = pd.read_csv(test_csv)
        target_vals = outdf.target.values
                # id_vals = df.id.values
        pred_vals = outdf.prediction.values

         
        mae = mean_absolute_error(target_vals, pred_vals)
        maes.append(mae)
                #task.record(fold, pred_vals)
          
        print(
                    "Dataset_name, Fold, MAE=",
                    
                    fold,
                    mean_absolute_error(target_vals, pred_vals),
                )
        print("Current working directory:", os.getcwd())
        os.chdir("..")
        print("Current working directory:", os.getcwd())
        print()
        print()
        print()
        print()
        print()
        print()
        print("+"*40)
        print("+"*40)

                #break



    maes = np.array(maes)
    print(maes, np.mean(maes), np.std(maes))
    print("+"*40)
    print("+"*40)

    

if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    #config = loadjson(config_template)
    #train_tasks(mb=mb, file_format="poscar")

    train_tasks(mb="1", file_format="poscar",length=-1)


    