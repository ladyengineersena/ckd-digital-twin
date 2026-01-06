#!/usr/bin/env python3
import argparse, os
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
import joblib
from src.preprocess.features import load_data, build_summary_features, build_sequence_data

def eval_lgbm(model_path, data_dir):
    meta, labs, meds, events = load_data(data_dir)
    X,y,fnames,pids = build_summary_features(meta, labs)
    model = joblib.load(model_path)
    probs = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, probs)
    print("LGBM AUC:", auc)

def eval_lstm(model_path, data_dir):
    meta, labs, meds, events = load_data(data_dir)
    seqs, tabs, pids = build_sequence_data(meta, labs, seq_len=90)
    # load model
    import torch
    from src.models.lstm_model import LSTMEGFR
    device = "cpu"
    model = LSTMEGFR(seq_input_dim=1, tab_input_dim=tabs.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    import torch
    seq_t = torch.tensor(seqs, dtype=torch.float32)
    tab_t = torch.tensor(tabs, dtype=torch.float32)
    with torch.no_grad():
        preds = model(seq_t, tab_t).numpy()
    # y_true: same naive target as training
    y_true = np.array([s[-1,0] for s in seqs])
    mse = mean_squared_error(y_true, preds)
    print("LSTM MSE:", mse)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lgbm","lstm"], default="lgbm")
    parser.add_argument("--model", default="outputs/lgbm_model.joblib")
    parser.add_argument("--data_dir", default="data/synthetic")
    args = parser.parse_args()
    if args.mode == "lgbm":
        eval_lgbm(args.model, args.data_dir)
    else:
        eval_lstm(args.model, args.data_dir)

