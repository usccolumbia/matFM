# Composition-Based MatBench Results

This directory contains the **composition-based** evaluation outputs of our ICL-FM framework on the MatBench benchmark suite. All experiments use **Magpie** or **MagpieEX** descriptors as inputs to the TabPFN-based in-context learning foundation model.

---

## Contents

### **1. JSON benchmark outputs (`*.json`)**
These files follow the official MatBench result format and include:
- task metadata  
- train/test split indices  
- predicted values  
- evaluation metrics (MAE)  

They can be loaded using the `matbench` Python API for verification and comparison.

---

## External Storage (Figshare)

All large related files—including:
- full model checkpoints  
- extracted embeddings  
- prediction CSVs  

are archived on **Figshare** for long-term access:

➡️ **https://figshare.com**  
*(The DOI link will be added once uploaded.)*

---

## Reproducibility Notes

- All results follow the official MatBench 5-fold splits.  
- Random seeds were fixed to ensure identical partitions across models.  
- The JSON files in this directory are fully compatible with the `matbench` tools.

For details on training and evaluation, see the top-level `README.md` in the repository.
