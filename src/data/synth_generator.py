#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from datetime import datetime, timedelta

def simulate_patient(i, days=365, seed=0):
    np.random.seed(seed + i)
    pid = f"P{i:04d}"
    age = int(np.random.randint(30, 85))
    sex = np.random.choice(["M","F"])
    bmi = float(np.round(np.random.normal(28,4),1))
    diabetes = int(np.random.rand() < 0.35)
    htn = int(np.random.rand() < 0.5)
    baseline_eGFR = float(np.clip(np.random.normal(45 - 0.2*age + (-10 if diabetes else 0), 8), 10, 90))
    # individual progression rate (mL/min/1.73m2 per year)
    decline_rate = np.random.normal(-3.5, 2.0)  # average -3.5 per year
    # variation: faster progression for diabetics
    if diabetes:
        decline_rate += np.random.normal(-1.5, 1.0)
    # meds: antihypertensive, ACEi/ARB use flag at baseline
    acei = int(np.random.rand() < 0.6)
    statin = int(np.random.rand() < 0.5)
    # simulate daily labs for 'days'
    labs = []
    meds = []
    events = []
    start = datetime(2020,1,1) + timedelta(days=i*2)
    # simulate eGFR with noise and occasional acute drops
    for d in range(days):
        date = start + timedelta(days=d)
        # trend: linear decline scaled to days
        eGFR_trend = baseline_eGFR + (decline_rate/365.0)*d
        # random acute hit with small probability
        if np.random.rand() < 0.005:
            hit = np.random.normal(-10,5)
        else:
            hit = 0.0
        noise = np.random.normal(0, 2.0)
        eGFR = max(3.0, eGFR_trend + hit + noise)
        creatinine = max(0.2, 186.0 / eGFR + np.random.normal(0,0.05))  # approximate conversion for demo
        labs.append({"patient_id": pid, "date": date.strftime("%Y-%m-%d"), "eGFR": round(eGFR,2), "creatinine": round(creatinine,3)})
        # med dosing snapshot: assume ACEi once daily if on
        if acei:
            meds.append({"patient_id": pid, "date": date.strftime("%Y-%m-%d"), "drug":"ACEi", "dose_mg": float(10.0 + np.random.normal(0,2))})
        # event: dialysis initiation if eGFR < 10 for 7 consecutive days (we'll check later)
    # determine events: dialysis entry if sustained low eGFR
    df_labs = pd.DataFrame(labs)
    df_labs['eGFR_roll7'] = df_labs['eGFR'].rolling(7,min_periods=1).mean()
    dialysis_dates = df_labs[df_labs['eGFR_roll7'] < 10]['date'].tolist()
    for dd in dialysis_dates:
        events.append({"patient_id": pid, "date": dd, "event":"dialysis_start"})
    meta = {"patient_id": pid, "age": age, "sex": sex, "bmi": bmi, "diabetes": diabetes, "htn": htn, "baseline_eGFR": round(baseline_eGFR,2), "acei": acei, "statin": statin, "decline_rate_per_year": round(decline_rate,3)}
    return meta, labs, meds, events

def generate(n_patients=500, days=365, out_dir="data/synthetic"):
    os.makedirs(out_dir, exist_ok=True)
    metas, labs_all, meds_all, events_all = [], [], [], []
    for i in range(n_patients):
        meta, labs, meds, events = simulate_patient(i, days=days, seed=1000)
        metas.append(meta)
        labs_all.extend(labs)
        meds_all.extend(meds)
        events_all.extend(events)
    pd.DataFrame(metas).to_csv(os.path.join(out_dir,"metadata.csv"), index=False)
    pd.DataFrame(labs_all).to_csv(os.path.join(out_dir,"labs.csv"), index=False)
    pd.DataFrame(meds_all).to_csv(os.path.join(out_dir,"meds.csv"), index=False)
    pd.DataFrame(events_all).to_csv(os.path.join(out_dir,"events.csv"), index=False)
    print(f"Generated {n_patients} patients synthetic dataset to {out_dir}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--out", type=str, default="data/synthetic")
    args = parser.parse_args()
    generate(n_patients=args.n, days=args.days, out_dir=args.out)

