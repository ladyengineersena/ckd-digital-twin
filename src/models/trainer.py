import os, joblib, numpy as np
from .xgb_model import train_lgbm, save_model, load_model
from .lstm_model import LSTMEGFR
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim, torch.nn as nn

def train_lgbm_wrapper(X,y,out_path,params=None):
    model = train_lgbm(X,y,params)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_model(model,out_path)
    return model

def train_lstm_wrapper(X_seq, X_tab, y_reg, out_path, epochs=20, batch=32, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMEGFR(seq_input_dim=X_seq.shape[2], tab_input_dim=X_tab.shape[1])
    model.to(device)
    ds = TensorDataset(torch.tensor(X_seq,dtype=torch.float32), torch.tensor(X_tab,dtype=torch.float32), torch.tensor(y_reg,dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        total = 0.0
        for seq, tab, lab in dl:
            seq, tab, lab = seq.to(device), tab.to(device), lab.to(device)
            pred = model(seq,tab)
            loss = loss_fn(pred, lab)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {ep+1}/{epochs} loss:{total/len(dl):.4f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return model

