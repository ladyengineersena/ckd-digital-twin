#!/usr/bin/env python3
import argparse, os
from src.data.synth_generator import generate
from src.preprocess.features import load_data, build_summary_features, build_sequence_data, scale_tabular
from src.models.trainer import train_lgbm_wrapper, train_lstm_wrapper
import joblib, numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lgbm","lstm"], default="lgbm")
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--out", default="outputs/lgbm_model.joblib")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.data_dir,"metadata.csv")):
        print("Generating synthetic data...")
        generate(n_patients=500, days=365, out_dir=args.data_dir)
    meta, labs, meds, events = load_data(args.data_dir)
    if args.mode == "lgbm":
        X, y, fns, pids = build_summary_features(meta, labs)
        model = train_lgbm_wrapper(X,y,args.out)
        joblib.dump(fns, os.path.join(os.path.dirname(args.out),"feature_names.joblib"))
    else:
        seqs, tabs, pids = build_sequence_data(meta, labs, seq_len=90)
        # train LSTM to predict next-day eGFR as regression target (use last actual+1 day from labs)
        # build y_reg as next eGFR after seq window; here we use simple shifting
        # For demo, set y_reg to last value (auto-regressive)
        y_reg = [s[-1,0] for s in seqs]
        X_tab = tabs
        X_seq = seqs
        model = train_lstm_wrapper(X_seq, X_tab, y_reg, args.out, epochs=args.epochs)
    print("Training finished")

if __name__=="__main__":
    main()

