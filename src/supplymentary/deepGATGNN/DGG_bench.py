
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

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY

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
                    task.dataset_name.split("_")[-1]
                    + "_"
                    + str(ii)
                )

                if not os.path.exists(fold_name):
                    os.makedirs(fold_name)

                # Define subfolders
                subfolders = ["train", "test"]
                for sub in subfolders:
                    path = os.path.join(fold_name, sub)
                    if not os.path.exists(path):
                        os.makedirs(path)
                # Function to write POSCAR files and targets.csv
                def write_data(df, folder_path):
                    targets_path = os.path.join(folder_path, "targets.csv")
                    with open(targets_path, "w") as f:
                        for jj, j in df.iterrows():
                            id = j.name
                            atoms = pmg_to_atoms(j.structure)
                            pos_name = os.path.join(folder_path, str(id))+ ".poscar"  # POSCAR file path
                            atoms.write_poscar(pos_name)
                            val = j[target]
                            line = f"{id},{val}\n"
                            f.write(line)
                
      
                  

                # Write files
                write_data(train_df, os.path.join(fold_name, "train"))
                write_data(test_df, os.path.join(fold_name, "test"))
                '''
                # ALIGNN requires the id_prop.csv file
                f = open("targets.csv", "w")
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
                    
                '''

                os.chdir(fold_name)


                model = "DEEP_GATGNN_demo" #DEEP_GATGNN_demo
                ####### training command #########
                traincmd = (
                    "python ../DeeperGATGNN/main.py"
                    +" --job_name='baseline' "
                    +" --run_mode='Training' "
                    +" --model="+model
                    +" --data_path=./train"
                    +" --format='"+file_format+"'"
                    +" --config_path='../matbench_config.yml' "
                    
                )
                print(traincmd)
                os.system(traincmd)

                print("+"*80)
                print()
                print()
                print()
                print()
                ###### testing command #########

                testcmd = (
                    "python ../DeeperGATGNN/main.py"
                    +" --run_mode='Predict' "
                    +" --model="+model
                    +" --data_path=./test"
                    +" --format='"+file_format+"'"
                    +" --config_path='../matbench_config.yml' "
                    
                )
                print(testcmd)
                os.system(testcmd)
                print("fold=",ii)
                print("+"*80)
                print()
                print()
                print()
                print()
                ###### feature extract command #########


                train_feat_cmd = (
                    "python ../DeeperGATGNN/main.py"
                    +" --run_mode='Analysis' "
                    +" --job_name='train_feat' "
                    +" --model="+model
                    +" --data_path=./train"
                    +" --format='"+file_format+"'"
                    +" --config_path='../matbench_config.yml' "
                    
                )
                print(train_feat_cmd)
                os.system(train_feat_cmd)

                test_feat_cmd = (
                    "python ../DeeperGATGNN/main.py"
                    +" --run_mode='Analysis' "
                    +" --job_name='test_feat' "
                    +" --model="+model
                    +" --data_path=./test"
                    +" --format='"+file_format+"'"
                    +" --config_path='../matbench_config.yml' "
                    
                )
                print(test_feat_cmd)
                os.system(test_feat_cmd)
                print("+"*80)
                print()
                print()
                print()
                print()


                test_csv =  "./testing_predicted_outputs.csv"
                df = pd.read_csv(test_csv)
                target_vals = df.target.values
                # id_vals = df.id.values
                pred_vals = df.prediction.values
                mae = mean_absolute_error(target_vals, pred_vals)
                maes.append(mae)
                task.record(fold, pred_vals)
          
                print(
                    "Dataset_name, Fold, MAE=",
                    task.dataset_name,
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

            task_name = task.dataset_name.split("_")[-1]
            mb.to_file(f"DGG_baseline_{task_name}.json")

if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    #config = loadjson(config_template)
    train_tasks(mb=mb, file_format="poscar")

    