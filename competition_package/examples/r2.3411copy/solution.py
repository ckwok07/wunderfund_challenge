import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")
from utils import DataPoint, ScorerStepByStep


class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = 64      # raw32 + rm5_32
        self.output_dim = 32     # delta raw32
        self.hidden_size = 64
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

        # scoring state
        self.current_seq_ix = None
        self.rm_sum = None
        self.rm_window = []
        self.buf = []

        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location=DEVICE))

        self.to(DEVICE)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])  # output delta only


    @torch.no_grad()
    def predict(self, dp: DataPoint):
        """
        Scoring:
        dp.state is raw32 → build 64-dim (raw + rolling mean)
        model outputs DELTA, but scorer expects RAW → return raw + delta
        """
        raw = dp.state.astype(np.float32)

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
        if len(self.buf) > 64:
            self.buf.pop(0)

        if not dp.need_prediction:
            return None

        seq = np.stack(self.buf, axis=0)[-64:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        pred_delta = self.forward(x).cpu().numpy().reshape(-1)
        if np.mean(pred_delta) < 0:
            pred_delta = -pred_delta
        pred_raw = raw + pred_delta
                # convert delta → raw prediction
        return pred_raw


# ---------- Training ----------
def make_sequences(arr64, seq_len=64):
    X, y = [], []
    for i in range(len(arr64) - seq_len):
        X.append(arr64[i:i+seq_len])
        curr = arr64[i+seq_len, :32]
        prev = arr64[i+seq_len - 1, :32]
        delta = curr - prev
        if np.mean(delta) < 0:       # enforce consistent sign
            delta = -delta
        y.append(delta)
     # delta target
    return np.stack(X), np.stack(y)


def amp_loss(pred, target, alpha=0.06):
    """MSE + amplitude-preserving variance penalty"""
    mse = torch.mean((pred - target) ** 2)
    var_pen = torch.mean((pred.std(0) - target.std(0)) ** 2)
    return mse + alpha * var_pen


def train_model(model, df, epochs=10, lr=1.4e-3, batch_size=96):
    raw = df.iloc[:, 3:35].values.astype(np.float32)
    rm5 = pd.DataFrame(raw).rolling(5, min_periods=1).mean().to_numpy().astype(np.float32)
    arr64 = np.concatenate([raw, rm5], axis=1).astype(np.float32)

    X_np, y_np = make_sequences(arr64, seq_len=64)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = amp_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{epochs}  loss={total / len(loader.dataset):.6f}")


# ---------- Main executable ----------
if __name__ == "__main__":
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
    scorer = ScorerStepByStep(f"{CURRENT_DIR}/../../datasets/train.parquet")

    if not os.path.exists(weights_path):
        print("No weights found — training...")
        model = PredictionModel()
        train_model(model, scorer.dataset)
        torch.save(model.state_dict(), weights_path)
        print("Saved weights → lstm_weights.pt\n")
    else:
        print("Weights found — skipping training.\n")

    model = PredictionModel()
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² = {results['mean_r2']:.6f}")
    for f in scorer.features:
        print(f"{f}: {results[f]:.6f}")
