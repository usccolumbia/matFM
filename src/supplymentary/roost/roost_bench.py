
import shutil
import glob
import os
from collections import defaultdict


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, roc_auc_score

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")
#condense_formula takes a material and returns the chemical formula in the correct format for CrabNet
def condense_formula(mat):
    if isinstance(mat, str):
        return mat
    else:
        return mat.formula.replace(' ', '')

#change_input runs condense_formula on all the input data used for training
def change_input(train_inputs):
  inputs = []
  for input in train_inputs:
    inputs.append(condense_formula(input))
  return inputs

def extract_number(mbid):
    """Extracts only the number from an mbid like 'mb-jdft2d-463'."""
    if isinstance(mbid, str):
        return mbid.split('-')[-1]
    return mbid

#make_df creates a data frame containing the train inputs and outputs for CrabNet
def make_df(train_inputs, train_outputs):

  input_df = pd.DataFrame({'composition': train_inputs, 'target': train_outputs})
  input_df.index = input_df.index.map(extract_number)
  #print(input_df.head())
  
  input_df.index.name = 'material_id' 
  #input_df['mbid'] = [extract_number(m) for m in mbid]
  return input_df
  

#make_df_test creates a data frame containing the test inputs for CrabNet
def make_df_test(test_inputs):
  test_df = pd.DataFrame({'composition' : test_inputs})
  #test_df.index = test_df.index.map(extract_number)
  test_df['target'] = np.nan
  test_df.index.name = 'material_id' 
  return test_df

#split_train_val splits the training data into two sets: training and validation
def split_train_val(df):
  df = df.sample(frac = 1.0, random_state = 7)
  val_df = df.sample(frac = 0.1, random_state = 7)
  train_df = df.drop(val_df.index)

  return train_df, val_df
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
import torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(666)

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
                #if ii <4:
                #    print("skipping fold", ii)
                #    continue
                train_inputs, train_outputs  = task.get_train_and_val_data(fold)
                test_inputs = task.get_test_data(fold,include_target=True, as_type="df")
                
                test_inputs.columns = ["composition","target"]
                test_inputs.index.name = 'material_id' 
                test_inputs["composition"] = test_inputs["composition"].apply(lambda x: condense_formula(x))
                #test_inputs['target'] = 0.0
                #print(test_inputs.head())

                fold_name = (
                    task.dataset_name.split("_")[-1]
                    + "_"
                    + str(ii)
                )
                data_dir = "./data/"+fold_name
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)

                fold_dir = "data/"+fold_name
                #Preparing the inputs data 
                inputs = change_input(train_inputs)
                
                df = make_df(inputs, train_outputs)
                df.to_csv(os.path.join(data_dir, 'train.csv'))

                test_inputs.to_csv(os.path.join(data_dir, 'test.csv'))

                #os.chdir(fold_name)

                #cmd = "conda run -n other_env python -c \"import sys; print(sys.executable)\""
                
                ####### training command ########
                print("starting training for fold", ii)
                traincmd = (
                    "conda run -n roost"
                    +" python -u ./roost/examples/roost-example.py"
                    +" --train"
                    +f" --data-path ./{fold_dir}/train.csv"
                    +" --val-size 0.1"
                    #+f" --val-path ./{fold_name}/val.csv"
                    +f" --test-path ./{fold_dir}/test.csv"
                    +" --targets target"
                    +f" --model-name {fold_name}"
                    +" --fea-path ./roost/data/el-embeddings/matscholar-embedding.json"
                   
                )
                #print(traincmd)
                #subprocess.run(traincmd, shell=True)# this is quiet
    
                os.system(traincmd)

                print("+"*80)
                print()
                print()
           
                ###### testing command #########

                testcmd = (
                        "conda run -n roost"
                        +" python ./roost/examples/roost-example.py"
                        +" --evaluate"
                        +f" --data-path ./{fold_dir}/test.csv"
                        +f" --test-path ./{fold_dir}/test.csv"
                        +" --targets target"
                        +f" --model-name {fold_name}"
                        +" --fea-path ./roost/data/el-embeddings/matscholar-embedding.json"
                )
                #print(testcmd)
                os.system(testcmd)


                print("fold=",ii)
                print("+"*80)
                print()
                print()
        
                ###### feature extract command #########


                train_feat_cmd = (
                        "conda run -n roost"
                        +" python ./roost/examples/roost-extract.py"
                        +" --evaluate"
                        +f" --data-path ./{fold_dir}/train.csv"
                        +f" --test-path ./{fold_dir}/train.csv"
                        +" --targets target"
                        +f" --model-name {fold_name}"
                        +" --fea-path ./roost/data/el-embeddings/matscholar-embedding.json"
                )
                test_feat_cmd = (
                        "conda run -n roost"
                        +" python ./roost/examples/roost-extract.py"
                        +" --evaluate"
                        +f" --data-path ./{fold_dir}/test.csv"
                        +f" --test-path ./{fold_dir}/test.csv"
                        +" --targets target"
                        +f" --model-name {fold_name}"
                        +" --fea-path ./roost/data/el-embeddings/matscholar-embedding.json"
                )


                #print(train_feat_cmd)
                os.system(train_feat_cmd)
                #print(test_feat_cmd)
                os.system(test_feat_cmd)


                print("+"*80)
                print()
                print()
     


                test_csv =  f"./results/{fold_name}.csv"
                # Add column names: mbid, target, pred
                col_names = ["mbid", "composition", "prediction","target"]

                df = pd.read_csv(test_csv,names=col_names, header=0)
                df = df.sort_values(by="mbid")  # ascending order by mbid
                df = df.reset_index(drop=True)   # reset the index after sorting

                target_vals = df.target.values
                #id_vals = df.mbid.values
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
                #os.chdir("..")
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
            mb.to_file(f"roost_baseline_{task_name}.json")

if __name__ == "__main__":
    config_template = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config_example.json")
    )
    #config = loadjson(config_template)
    train_tasks(mb=mb, file_format="poscar")

    