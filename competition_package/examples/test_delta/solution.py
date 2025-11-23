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

        self.dim = 32                         # raw feature dimension
        self.hidden_size = 64
        self.num_layers = 1
        self.max_seq_len = 32                 # history size

        # online state
        self.current_seq_ix = None
        self.sequence_history = []
        self.last_state = None                # previous raw for delta calc

        # delta model: input = raw, target = delta
        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.dim)

        # load weights
        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=DEVICE)
            self.load_state_dict(state_dict)

        # load amplitude correction stats
        self.bias = None
        self.gain = None
        bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")
        gain_path = os.path.join(CURRENT_DIR, "delta_gain.npy")
        if os.path.exists(bias_path):
            self.bias = np.load(bias_path).astype(np.float32)
        if os.path.exists(gain_path):
            self.gain = np.load(gain_path).astype(np.float32)

        self.to(DEVICE)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        # reset for new sequence
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.sequence_history = []
            self.last_state = None

        raw_state = dp.state.astype(np.float32)

        # delta relative to last
        if self.last_state is None:
            delta = np.zeros_like(raw_state)
        else:
            delta = raw_state - self.last_state
        self.last_state = raw_state

        # add raw to history (model predicts delta)
        self.sequence_history.append(raw_state)
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        # only predict when scorer requests
        if not dp.need_prediction:
            return None

        seq = np.stack(self.sequence_history, axis=0)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        delta_pred = self.forward(x).detach().cpu().numpy().reshape(-1)

        # amplitude correction
        if self.bias is not None:
            delta_pred -= self.bias
        if self.gain is not None:
            delta_pred *= self.gain

        return raw_state + delta_pred


def make_sequences(raw: np.ndarray, seq_len: int = 32):
    X, y = [], []
    for i in range(len(raw) - seq_len):
        X.append(raw[i:i+seq_len])
        y.append(raw[i+seq_len] - raw[i+seq_len-1])    # delta target
    return np.stack(X), np.stack(y)


def train_model(
    model: PredictionModel,
    train_raw: np.ndarray,
    num_epochs: int = 3,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    model.train()
    model.to(DEVICE)

    X, y = make_sequences(train_raw, seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{num_epochs} - loss={total/len(loader.dataset):.6f}")

    # compute bias + gain for amplitude fix
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            delta_pred = model(xb).cpu().numpy()
            preds_all.append(delta_pred)
            trues_all.append(yb.numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)

    bias = (preds_all - trues_all).mean(axis=0)
    std_pred = preds_all.std(axis=0) + 1e-6
    std_true = trues_all.std(axis=0) + 1e-6
    gain = std_true / std_pred

    np.save(os.path.join(CURRENT_DIR, "delta_bias.npy"), bias)
    np.save(os.path.join(CURRENT_DIR, "delta_gain.npy"), gain)


if __name__ == "__main__":
    scorer = ScorerStepByStep(f"{CURRENT_DIR}/../../datasets/train.parquet")
    train_raw = scorer.dataset[scorer.features].values.astype(np.float32)

    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")

    if os.path.exists(weights_path):
        print("⚡ Weights found — skipping training.")
        model = PredictionModel()
    else:
        print("🚀 No weights — training model.")
        model = PredictionModel()
        train_model(model, train_raw, num_epochs=3, lr=1e-3, batch_size=64)
        torch.save(model.state_dict(), weights_path)
        model = PredictionModel()

    model.eval()
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    for f in scorer.features[:5]:
        print(f"  {f}: {results[f]:.6f}")
    print("="*60)
