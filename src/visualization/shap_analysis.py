from tabpfn_extensions import TabPFNClassifier, interpretability
import torch
import pandas as pd
import numpy as np
import os
from tabpfn_extensions.embedding import TabPFNEmbedding


compo = "./Thermo_4"
ii = compo.split("_")[-1]
TF_feat = compo+"/TF_train_features_with_labels.pt"
TF_test = compo+f"/TF_test_features_with_labels.pt"
TF_tab = compo+f"/TF_tabpfn_model_{ii}.pt"
prefix = "500_f4_"

FMmodel = torch.load(TF_tab,weights_only=False)
FM_extractor = TabPFNEmbedding(tabpfn_clf=FMmodel, n_fold=0)
TF_dict = torch.load(TF_test)  # 每个键是 jid，值是 {latent, target}         
TFX = []
TFy =[]
for k, v in TF_dict.items():
    TFX.append(v['latent'].numpy() if torch.is_tensor(v['latent']) else np.array(v['latent']))
    TFy.append(v['target'])


TFX = np.stack(TFX).astype(np.float32)
TFy = np.array(TFy).astype(np.float32)
print(TFX.shape)
print(TFy.shape)

# Calculate SHAP values
shap_values = interpretability.shap.get_shap_values(
    estimator=FMmodel,
    test_x=TFX,
    algorithm="permutation",
    max_evals=1600,
)


save_shap_path =  f"./thermo_FFFM_fold4_shap.npy"
np.save(save_shap_path, shap_values)