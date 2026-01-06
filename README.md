# CKD Digital Twin — Chronic Kidney Disease Digital Twin (Research Prototype)

This repository provides a research-ready **digital twin** prototype for modelling CKD progression.

It includes:

- Synthetic data generator (patient demographics, labs, meds, eGFR time series)

- Feature engineering & preprocessing

- Baseline LightGBM model (tabular summary features)

- LSTM multimodal time-series model (sequence forecasting)

- Digital twin simulator that uses the LSTM to project counterfactual trajectories under interventions

- FastAPI inference endpoint (loads baseline model for quick scoring)

- Dockerfile and docker-compose

**Important:** For research/demo only. Not for clinical use. Use real patient data only after IRB approvals and appropriate governance.

Quickstart (local):

```bash

# 1. create venv and install

python -m venv .venv

# linux/mac

source .venv/bin/activate

# windows (PowerShell)

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 2. Generate synthetic data

python -m src.data.synth_generator --n 500 --out data/synthetic

# 3. Train baseline LightGBM

python -m src.pipelines.train_pipeline --mode lgbm --data_dir data/synthetic --out outputs/lgbm_model.joblib

# 4. Train LSTM (sequence model)

python -m src.pipelines.train_pipeline --mode lstm --data_dir data/synthetic --out outputs/lstm_model.pt --epochs 20

# 5. Run simulator demo (from notebooks or script)

python -m src.simulator.twin --data_dir data/synthetic --model outputs/lstm_model.pt --patient P0001 --scenario blood_pressure_control

# 6. Run API

uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

