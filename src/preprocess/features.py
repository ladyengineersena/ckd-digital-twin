import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(data_dir="data/synthetic"):
    meta = pd.read_csv(os.path.join(data_dir,"metadata.csv"))
    labs = pd.read_csv(os.path.join(data_dir,"labs.csv"), parse_dates=["date"])
    meds = pd.read_csv(os.path.join(data_dir,"meds.csv"), parse_dates=["date"])
    events = pd.read_csv(os.path.join(data_dir,"events.csv")) if os.path.exists(os.path.join(data_dir,"events.csv")) else pd.DataFrame()
    return meta, labs, meds, events

def build_summary_features(meta, labs, window_days=30):
    # compute per-patient summary features from last window_days of labs
    patients = meta["patient_id"].values
    feats = []
    labels = []
    pids = []
    for pid in patients:
        sub = labs[labs["patient_id"]==pid].sort_values("date")
        if sub.empty:
            # default zeros
            mean_egfr = 0.0; sd_egfr = 0.0; last_egfr = 0.0; slope = 0.0
        else:
            last_date = sub["date"].max()
            window = sub[sub["date"] >= (last_date - pd.Timedelta(days=window_days))]
            if window.empty:
                window = sub.tail(1)
            mean_egfr = float(window["eGFR"].mean())
            sd_egfr = float(window["eGFR"].std() if window["eGFR"].std() is not None else 0.0)
            last_egfr = float(sub.iloc[-1]["eGFR"])
            # slope (linear fit on full series)
            if len(sub) >= 2:
                x = (sub["date"] - sub["date"].min()).dt.days.values
                y = sub["eGFR"].values
                slope = float(np.polyfit(x, y, 1)[0])
            else:
                slope = 0.0
        meta_row = meta[meta["patient_id"]==pid].iloc[0]
        age = float(meta_row["age"])
        diabetes = int(meta_row["diabetes"])
        htn = int(meta_row["htn"])
        acei = int(meta_row["acei"])
        # label for event: dialysis in events?
        had_dialysis = False
        # naive label from events dataframe omitted; rely on events file if available
        feats.append([mean_egfr, sd_egfr, last_egfr, slope, age, diabetes, htn, acei])
        labels.append(int(last_egfr < 15) if not sub.empty else 0)
        pids.append(pid)
    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=int)
    feature_names = ["mean_egfr_30d","sd_egfr_30d","last_egfr","slope_egfr","age","diabetes","htn","acei"]
    return X, y, feature_names, pids

def build_sequence_data(meta, labs, seq_len=90, feature_cols=["eGFR"]):
    # For each patient, produce a fixed-length sequence of last seq_len days of feature_cols
    patients = meta["patient_id"].values
    seqs = []
    tabs = []
    pids = []
    for pid in patients:
        sub = labs[labs["patient_id"]==pid].sort_values("date")
        values = sub[feature_cols].values if not sub.empty else np.zeros((0,len(feature_cols)))
        # take last seq_len days: pad with last observation if short
        if values.shape[0] >= seq_len:
            seq = values[-seq_len:,:]
        elif values.shape[0] == 0:
            seq = np.zeros((seq_len, len(feature_cols)))
        else:
            pad = np.tile(values[-1,:], (seq_len - values.shape[0],1))
            seq = np.vstack([pad, values])
        # tab features
        meta_row = meta[meta["patient_id"]==pid].iloc[0]
        tab = [meta_row["age"], meta_row["bmi"], meta_row["diabetes"], meta_row["htn"], meta_row["acei"]]
        seqs.append(seq)
        tabs.append(tab)
        pids.append(pid)
    return np.array(seqs, dtype=float), np.array(tabs, dtype=float), pids

def scale_tabular(X_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    return Xs, scaler

