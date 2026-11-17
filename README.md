# Deep Learning-Based Load Forecasting for Microgrid Operation Optimization

**Graduation thesis:** *A Study on the Application of Deep Learning Models to Microgrid Optimization Problems*  
**Author:** Do Quynh Huong  
**Institution:** Hanoi University of Science and Technology (HUST), School of Electrical and Electronic Engineering  
**Supervisor:** Assoc. Prof. Dr. Bui Dang Thanh  
**Year:** 2026

## Overview

This repository contains the Jupyter notebooks and supporting code for a graduation thesis on **short-term electrical load forecasting** and its application to **microgrid operation optimization**.

The study develops a **CNN–BiLSTM–Attention** model for one-hour-ahead forecasting of an aggregated campus load, compares it with LSTM, GRU, BiLSTM, and CNN–LSTM baselines, and uses the forecasts in a **linear programming (LP)** model of microgrid energy dispatch.

The optimization model includes solar photovoltaic (PV) generation, battery energy storage (BESS), a diesel generator, and electricity purchases/sales with the utility grid under time-of-use tariffs.

### Research workflow

```text
Building Data Genome Project 2 (BDG2)
           |
           v
Campus load aggregation + weather integration
           |
           v
Feature engineering (37 features) + time-series preprocessing
           |
           v
One-hour-ahead load forecasting
    |                 |
    |                 +-- Proposed CNN–BiLSTM–Attention
    +-- LSTM / GRU / BiLSTM / CNN–LSTM
           |
           v
Model evaluation and comparison
           |
           v
24-hour LP-based microgrid scheduling
           |
           v
Operational and economic comparison
```

## Methodology

- **Data:** Hourly measurements from 2016–2017 for seven educational buildings at the **Robin** site in BDG2. Building electricity loads are aggregated to form a campus-scale load series, then joined with site weather measurements.
- **Features:** 37 inputs/columns, including load, weather, cyclical time encodings, lagged loads, rolling statistics, differences, and load-related ratios.
- **Forecasting task:** 48 hours of historical observations (48 × 37) are used to predict the next hour's aggregated electrical load.
- **Data split:** Chronological train/validation/test split of approximately 70%/15%/15%; the `MinMaxScaler` is fitted on training data only.
- **Models:** LSTM, GRU, BiLSTM, CNN–LSTM, and a proposed network combining dilated 1D convolution, residual connections, stacked BiLSTMs, and an attention mechanism.
- **Evaluation:** MAE, RMSE, R², and the repository's daily-aggregated MAPE calculation (see `helpers.py`).
- **Optimization:** A 24-hour LP implemented using `scipy.optimize.linprog`, with dispatch and battery state-of-charge constraints and time-of-use energy prices.

## Repository contents

| File | Description |
| --- | --- |
| `01_EDA_DataPreprocessing.ipynb` | Dataset exploration, campus load construction, feature engineering, scaling, and sequence creation. |
| `02_Baseline_Models.ipynb` | Train and evaluate LSTM and GRU. |
| `03_Advanced_Models.ipynb` | Train and evaluate BiLSTM and CNN–LSTM. |
| `04_Proposed_Model.ipynb` | Build, train, and analyze CNN–BiLSTM–Attention. |
| `05_Model_Comparison.ipynb` | Compare saved model predictions and generate evaluation figures. |
| `06_Microgrid_Optimization.ipynb` | Optimize 24-hour microgrid operation and compare operating costs. |
| `helpers.py` | Shared inverse-scaling, metric, and plotting utilities. |
| `data/config.json` | Feature list, forecasting setup, and dataset configuration. |
| `data/README.md` | Dataset source, required input files, and preprocessing outputs. |
| `doc/` | Thesis report and presentation, if included. |

**Note:** `data/` contains large files when running locally; `models/` and `results/` are generated during notebook execution. They are not required to browse the notebooks on GitHub.

## Installation

Use a Python environment with a compatible TensorFlow installation (Python 3.11 is a practical starting point). From the **repository root**:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

The versions in `requirements.txt` describe a compatible starting environment, **not an independently validated reproduction of the original training environment**.

## Data setup

The original large data files are **not included** in the default public repository. Obtain them from the [official BDG2 project](https://github.com/buds-lab/building-data-genome-project-2), then place these files in `data/` with the exact names below:

```text
data/
├── README.md
├── config.json
├── electricity_cleaned.csv   # required to run Notebook 01
├── metadata.csv              # required to run Notebook 01
└── weather.csv               # required to run Notebook 01
```

See [data/README.md](data/README.md) for the original source locations and generated artifacts. The files must have the expected columns and layout used by Notebook 01.

## Running the notebooks

Open Jupyter Lab **from the repository root** (the notebooks resolve data paths relative to the current working directory), then execute the notebooks in numerical order:

1. `01_EDA_DataPreprocessing.ipynb` — creates `data/processed_data.npz`, `data/scaler.pkl`, the configuration, and intermediate CSVs.
2. `02_Baseline_Models.ipynb` — trains baselines and writes `results/baseline_predictions.npz`.
3. `03_Advanced_Models.ipynb` — trains additional models and writes `results/advanced_predictions.npz`.
4. `04_Proposed_Model.ipynb` — trains the proposed architecture and writes `results/proposed_predictions.npz`.
5. `05_Model_Comparison.ipynb` — reads saved predictions and creates comparison tables and plots.
6. `06_Microgrid_Optimization.ipynb` — reads saved predictions and performs LP-based dispatch analysis.

Training requires significantly more time than reviewing the notebooks; a compatible GPU can be helpful but is not necessary to read the project. The notebooks save model checkpoints under `models/` and figures/metrics under `results/`.

## Results and thesis

The notebooks contain model-building procedures, saved execution outputs, evaluation plots, and microgrid optimization analyses. In particular:

- **Notebook 05:** forecasting model comparison and prediction visualizations.
- **Notebook 06:** battery scheduling, energy dispatch, cost analysis, and comparisons with a perfect-forecast reference.

For the full research background, methodology, experiments, and discussion, see the thesis documents if uploaded:

- [Graduation thesis report](doc/DO-QUYNH-HUONG-20222779-GRAD-THESIS-REPORTpdf2.pdf)
- [Thesis presentation](doc/DO-QUYNH-HUONG-Slide-Graduation-thesis.pptx)

The links above assume the uploaded documents retain their original filenames inside `doc/`.

## Notes on reproducibility

- Raw BDG2 datasets, intermediate arrays, checkpoints, and generated result files are omitted from the default Git upload; rerun the notebooks in order to produce them locally.
- Notebook 05 and Notebook 06 depend on prediction archives saved by Notebooks 02–04.
- The `MAPE` function in `helpers.py` computes percentage error from **24-hour block averages** when at least one complete block is available. It should not be interpreted as standard pointwise hourly MAPE.
- The operational model is a research simulation; its PV profile, tariffs, capacities, and other operating assumptions are defined in Notebook 06.

## Data attribution

Data originate from the **Building Data Genome Project 2 (BDG2)**. Please cite the dataset creators when using this project:

Miller, C., Kathirgamanathan, A., Picchetti, B., et al. (2020). *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition.* **Scientific Data, 7**, Article 368. https://doi.org/10.1038/s41597-020-00712-x

## License

The original source code in this repository is offered under the [MIT License](LICENSE). Third-party datasets, referenced publications, figures, and thesis materials are not automatically relicensed by this code license; consult their respective sources and rights notices.
