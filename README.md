# matFM

## In context learning Foundation models for Materials Property Prediction with Small datasets

ICL-FM is a unified in-context learning framework for materials property prediction that combines TabPFN, Magpie descriptors, and GNN-based structural embeddings. It enables fast, training-free inference with strong performance on MatBench and thermal conductivity benchmarks, while offering interpretable, physics-aware feature refinement.

## Features
- Competitive Performance Across Benchmarks
- Fast, Training-Free Inference
- Physics-Aware Interpretability

## Repository Structure
```
matFM/
├── README.md 
├── LICENSE 
├── environments/ 
│ ├── alignn.yml
│ └── tabpfn.yml
├── results/ 
│ ├── figures/ # Plots and visualizations
│ ├── matbench/ # MatBench benchmark outputs
│ └── thermo/ # Lattice thermal conductivity results
└── src/ # Source code
  ├── matbench/ # MatBench experiments and data loaders
  ├── supplementary/ # Additional scripts & ablation utilities
  ├── thermo/ # Thermal conductivity dataset experiments
  └── visualization/ # Plotting and analysis tools
```

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/your-username/matFM.git
cd matFM
```
---

### 2. Set up the environment

Two environment files are provided depending on the components you wish to use.

#### TabPFN environment
For running the in-context learning foundation model (ICL-FM):
```bash
conda env create -f environments/tabpfn.yml
conda activate tabpfn
```

#### ALIGNN environment 
For generating structure-aware GNN embeddings:

```bash
conda env create -f environments/alignn.yml
conda activate alignn
```
#### Other environment 
For other models listed in MatBench, please follow their respective installation instructions.
To ensure a smooth pipeline and avoid package conflicts, each model should use its own dedicated Conda environment.
Our scripts are designed to support running different models across multiple environments.

---

### 3. (Optional) Install the package locally
```bash
pip install -e .
```

---

### 4. Verify the installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```
## Usage Examples


Example command for running the Magpie + TabPFN evaluation on a MatBench task:

```bash
python src/matbench/magpie/magpie_matbench.py
```

By default, `magpie_matbench.py` evaluates the model on the **Matbench JDFT-2D** task.  
Users can modify the dataset by editing the `MatbenchBenchmark` configuration inside:
```
src/matbench/magpie/magpie_matbench.py
```

Example configuration:

```python
MatbenchBenchmark(
    autoload=False,
    subset=[
        "matbench_jdft2d",      # Default dataset
        # "matbench_phonons",
        # "matbench_dielectric",
        # "matbench_log_gvrh",
        # "matbench_log_kvrh",
        # "matbench_perovskites",

    ],
)
```

Uncomment any dataset in the list to run ICL-FM on that benchmark.

## Results

The `results/` directory contains all outputs generated during the experiments:

```
results/
│── figures/      # Plots and visualizations used in the manuscript
│── matbench/     # MatBench benchmark prediction outputs (JSON)
└── thermo/       # Lattice thermal conductivity prediction results (CSV)
```

Trained model checkpoints are **available upon request** due to their large file size.

## Citation
If you use **matFM** or the **ICL-FM** framework in your research, please cite the following:

```bibtex
@article{matfm2025icl,
  title     = {In context learning Foundation models for Material Property Prediction with Small datasets},
  author    = {Qinyang Li, Rongzhi Dong, Jianjun Hu},
  journal   = {Preprint},
  year      = {2025},
  url       = {https://github.com/usccolumbia/matFM}
}
```