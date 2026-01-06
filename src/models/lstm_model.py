import torch
import torch.nn as nn

class LSTMEGFR(nn.Module):
    def __init__(self, seq_input_dim=1, seq_hidden=64, tab_input_dim=5, tab_hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_input_dim, hidden_size=seq_hidden, num_layers=1, batch_first=True)
        self.tab_fc = nn.Sequential(nn.Linear(tab_input_dim, tab_hidden), nn.ReLU(), nn.Dropout(0.2))
        self.fc = nn.Sequential(nn.Linear(seq_hidden + tab_hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,1))
    def forward(self, seq, tab):
        # seq: (batch, timesteps, features)
        out, (h,c) = self.lstm(seq)
        seq_emb = out[:, -1, :]
        tab_emb = self.tab_fc(tab)
        x = torch.cat([seq_emb, tab_emb], dim=1)
        return self.fc(x).squeeze(-1)  # predict next eGFR (regression)

