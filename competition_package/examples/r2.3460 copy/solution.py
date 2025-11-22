import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", DEVICE)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")
from utils import DataPoint, ScorerStepByStep


class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_dim = 64      # raw + rm5
        self.output_dim = 32
        self.hidden_size = 64
        self.num_layers = 1

        # LSTM accepts 64-dim during training + scoring
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

        # state for scoring
        self.current_seq_ix = None
        self.rm_sum = None
        self.rm_window = []

        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location=DEVICE))

        self.to(DEVICE)

    # forward for training + scoring (expects seq64)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


    @torch.no_grad()
    def predict(self, dp: DataPoint):
        """Scoring: dp.state is 32-dim → build 64-dim (raw + rm5) → predict"""
        raw = dp.state.astype(np.float32)

        # new sequence
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.rm_sum = raw.copy()
            self.rm_window = [raw]
            self.buf = []

        else:
            self.rm_sum += raw
            self.rm_window.append(raw)
            if len(self.rm_window) > 5:
                self.rm_sum -= self.rm_window.pop(0)

        rm = self.rm_sum / len(self.rm_window)
        x64 = np.concatenate([raw, rm]).astype(np.float32)
        self.buf.append(x64)

        # store sequence history
        if len(self.buf) > 64:
            self.buf.pop(0)
        if not dp.need_prediction:
            return None

        seq = np.stack(self.buf, axis=0)[-64:]  # last 64 timesteps
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        y = self.forward(x)
        return y.cpu().numpy().reshape(-1)


def make_sequences(data, seq_len=64):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, :32])   # next raw32
    return np.stack(X), np.stack(y)


def train_model(model, df, num_epochs=10, lr=1.4e-3, batch_size=96):
    raw = df.iloc[:, 3:35].values.astype(np.float32)
    rm5 = pd.DataFrame(raw).rolling(5, min_periods=1).mean().to_numpy().astype(np.float32)
    arr64 = np.concatenate([raw, rm5], axis=1).astype(np.float32)

    X_np, y_np = make_sequences(arr64, seq_len=64)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False)

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
        print(f"Epoch {epoch + 1}/{num_epochs}  loss={total/len(loader.dataset):.6f}")


if __name__ == "__main__":
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
    scorer = ScorerStepByStep(f"{CURRENT_DIR}/../../datasets/train.parquet")

    # Train only once
    if not os.path.exists(weights_path):
        print("No weights found → training...")
        model = PredictionModel()
        train_model(model, scorer.dataset)
        torch.save(model.state_dict(), weights_path)
        print("Weights saved → lstm_weights.pt\n")

    else:
        print("Weights found → skipping training.\n")

    # Score
    model = PredictionModel()
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² = {results['mean_r2']:.6f}")
    for i in range(len(scorer.features)):
        print(f"{scorer.features[i]}: {results[scorer.features[i]]:.6f}")
