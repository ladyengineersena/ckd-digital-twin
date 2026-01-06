import os, pandas as pd

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)

def to_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

