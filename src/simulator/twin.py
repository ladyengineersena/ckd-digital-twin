#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from src.preprocess.features import load_data, build_sequence_data
from src.models.lstm_model import LSTMEGFR
import torch
from src.simulator.interventions import apply_intervention_sequence

def load_model(model_path, device):
    model = LSTMEGFR(seq_input_dim=1, tab_input_dim=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def forecast_patient(patient_id, model, X_seq, X_tab, pids, horizon=90, device="cpu", intervention=None):
    # find index
    idx = pids.index(patient_id) if patient_id in pids else None
    if idx is None:
        raise ValueError("patient not found")
    seq = X_seq[idx:idx+1,:,:]  # shape (1, seq_len, features)
    tab = X_tab[idx:idx+1,:]
    seq_copy = seq.copy()
    preds = []
    model.to(device)
    with torch.no_grad():
        for t in range(horizon):
            seq_t = torch.tensor(seq_copy, dtype=torch.float32).to(device)
            tab_t = torch.tensor(tab, dtype=torch.float32).to(device)
            pred = model(seq_t, tab_t).cpu().numpy()[0]
            preds.append(float(pred))
            # append pred and drop first -> rolling
            next_row = np.array([[pred]])
            seq_copy = np.concatenate([seq_copy[:,1:,:], next_row.reshape(1,1,1)], axis=1)
            # optionally apply simple intervention to tab (e.g., set acei to 1)
            if intervention == "start_acei":
                tab[0,-1] = 1  # assuming last tab field is acei
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--model", required=True)
    parser.add_argument("--patient", default="P0000")
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--intervention", default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, labs, meds, events = load_data(args.data_dir)
    X_seq, X_tab, pids = build_sequence_data(meta, labs, seq_len=90)
    model = load_model(args.model, device)
    preds = forecast_patient(args.patient, model, X_seq, X_tab, list(pids), horizon=args.horizon, device=device, intervention=args.intervention)
    df = pd.DataFrame({"day": list(range(1,len(preds)+1)), "pred_eGFR": preds})
    print(df.head())
    outp = os.path.join("outputs", f"forecast_{args.patient}.csv")
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(outp, index=False)
    print("Saved forecast to", outp)

if __name__=="__main__":
    main()

