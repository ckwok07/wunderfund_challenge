import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", DEVICE)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")
from utils import DataPoint, ScorerStepByStep


class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_dim = 64      # raw32 + rm32
        self.output_dim = 32
        self.hidden_size = 64
        self.num_layers = 1

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Linear head
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

        # seq buffer (scoring only)
        self.current_seq_ix = None
        self.max_len = 600
        self.buffer = torch.empty(1, self.max_len, self.input_dim, device=DEVICE)
        self.seq_len = 0

        # load pretrained weights
        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            print("Loading existing lstm_weights.pt")
            self.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        self.to(DEVICE)

    # FOR TRAINING (xb is 64-dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

    # FOR SCORING (dp is 32-dim)
    @torch.no_grad()
    def predict(self, dp: DataPoint):

        # reset if new timeseries
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.seq_len = 0
            self.rm_sum = None
            self.rm_window = []

        s = dp.state.astype(np.float32)

        # rolling mean (5)
        if self.rm_sum is None:
            self.rm_sum = s.copy()
            self.rm_window = [s]
            rm = s
        else:
            self.rm_sum += s
            self.rm_window.append(s)
            if len(self.rm_window) > 5:
                self.rm_sum -= self.rm_window.pop(0)
            rm = self.rm_sum / len(self.rm_window)

        # build 64-dim input
        inp = np.concatenate([s, rm]).astype(np.float32)
        t = torch.from_numpy(inp).to(DEVICE)

        # append into sliding buffer
        if self.seq_len < self.max_len:
            self.buffer[0, self.seq_len].copy_(t)
            self.seq_len += 1
        else:
            self.buffer[:, :-1] = self.buffer[:, 1:].clone()
            self.buffer[0, -1].copy_(t)

        if not dp.need_prediction:
            return None

        x = self.buffer[:, :self.seq_len]
        out, _ = self.lstm(x)
        y = self.fc(out[:, -1])
        return y.cpu().numpy().reshape(-1)


def make_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, :32])
    return np.stack(X), np.stack(y)


def train_model(model, train_array, num_epochs=4, seq_len=64, lr=1.5e-3, batch_size=64):
    model.train()
    X, y = make_sequences(train_array, seq_len=seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y),
                                         batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{num_epochs} loss={total/len(loader.dataset):.6f}")


if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    scorer = ScorerStepByStep(test_file)
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")

    # train only once
    if not os.path.exists(weights_path):
        print("No model weights found → training...")
        df = scorer.dataset.copy()
        raw = df.iloc[:, 3:35]      # 32 raw features
        rm5 = raw.rolling(window=5, min_periods=1).mean()
        train_array = np.concatenate([raw.values, rm5.values], axis=1).astype(np.float32)

        model = PredictionModel()
        train_model(model, train_array, num_epochs=4, seq_len=64, lr=1.5e-3)
        torch.save(model.state_dict(), weights_path)
        print("Saved lstm_weights.pt")
    else:
        print("Using existing lstm_weights.pt")

    # score
    model = PredictionModel()
    results = scorer.score(model)

    print("\nResults")
    print(f"Mean R² = {results['mean_r2']:.6f}")
    for i in range(len(scorer.features)):
        print(f"{i}: {results[scorer.features[i]]:.6f}")
