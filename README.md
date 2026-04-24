# EECE 693 Project

This repository contains the EECE 693 project pipeline for predicting asthma exacerbation risk from AAMOS-00 smartwatch data. The current workflow focuses on smartwatch-only signals, including heart rate, activity type, intensity, and steps.

## Repository Structure

```text
.
├── data/
│   ├── raw/          # Original AAMOS-00 input files used by the notebooks
│   ├── interim/      # Cleaned intermediate smartwatch data
│   └── processed/    # Feature tables and labeled modeling datasets
├── notebooks/        # Analysis, preprocessing, feature engineering, and modeling notebooks
├── outputs/
│   ├── figures/      # Generated plots
│   └── tables/       # Generated metrics and summaries
├── report/           # LaTeX progress report
├── src/              # Shared project paths and helper modules
├── requirements.txt
└── 693_proposal.pdf
```

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The virtual environment is intentionally ignored by git and should not be committed.

## Notebook Workflow

Run the notebooks in order:

1. `notebooks/01_data_understanding.ipynb`
2. `notebooks/02_smartwatch_cleaning.ipynb`
3. `notebooks/03_windowing_and_features.ipynb`
4. `notebooks/04_label_preparation.ipynb`
5. `notebooks/05_baseline_model_local_or_colab.ipynb`
6. `notebooks/06_improved_baselines_tier1.ipynb`
7. `notebooks/07_deep_learning_tier2.ipynb`

Notebook 07 is currently present as a placeholder for the Tier-2 deep learning work.

## Data and Outputs

The repository currently includes the raw, interim, processed, and output files needed to reproduce the current project state. Generated Python caches, notebook checkpoints, virtual environments, local environment files, and editor settings are excluded through `.gitignore`.

## Main Dependencies

- `pandas`, `numpy`, and `pyarrow` for data loading, feature tables, and Parquet files
- `matplotlib` for plots
- `scikit-learn` for tabular baselines and evaluation
- `xgboost` for the XGBoost baseline
- `tqdm` for progress bars
- `jupyterlab` and `ipykernel` for notebook execution
