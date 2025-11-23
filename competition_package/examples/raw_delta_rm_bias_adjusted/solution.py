import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", DEVICE)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint, ScorerStepByStep


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.raw_dim = 32
        self.input_dim = 96     # [raw, delta, rolling mean]
        self.output_dim = 32    # predict next delta (32 dims)

        self.hidden_size = 96
        self.num_layers = 2
        self.max_seq_len = 32
        self.rm_window = 5

        self.current_seq_ix = None
        self.sequence_history = []
        self.last_state = None
        self.delta_buffer = []

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.15 if self.num_layers > 1 else 0.0
        )

        self.fc_delta = nn.Linear(self.hidden_size, self.output_dim)
        self.fc_raw = nn.Linear(self.hidden_size, self.raw_dim)

        self.weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        self.bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")

        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}")
            state_dict = torch.load(self.weights_path, map_location=DEVICE)
            self.load_state_dict(state_dict)

        if os.path.exists(self.bias_path):
            print(f"Loading bias from {self.bias_path}")
            self.bias = np.load(self.bias_path).astype(np.float32)
        else:
            self.bias = None

        self.to(DEVICE)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]

        delta_out = self.fc_delta(last)
        raw_out = self.fc_raw(last)
        return delta_out, raw_out

    def _reset_sequence_state(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.sequence_history = []
        self.last_state = None
        self.delta_buffer = []

    def _update_rolling_mean(self, new_delta: np.ndarray) -> np.ndarray:
        self.delta_buffer.append(new_delta)
        if len(self.delta_buffer) > self.rm_window:
            self.delta_buffer = self.delta_buffer[-self.rm_window:]
        stacked = np.stack(self.delta_buffer, axis=0)
        return stacked.mean(axis=0)

    @torch.no_grad()
    def predict(self, dp: DataPoint):

        if self.current_seq_ix != dp.seq_ix:
            self._reset_sequence_state(dp.seq_ix)

        raw_state = dp.state.astype(np.float32)

        if self.last_state is None:
            delta = np.zeros_like(raw_state, dtype=np.float32)
        else:
            delta = raw_state - self.last_state

        self.last_state = raw_state
        rm = self._update_rolling_mean(delta)

        feat = np.concatenate([raw_state, delta, rm], axis=0)
        self.sequence_history.append(feat)
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        if not dp.need_prediction:
            return None

        seq = np.stack(self.sequence_history, axis=0)
        x = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        delta_pred, _ = self.forward(x)
        delta_pred_np = delta_pred.detach().cpu().numpy().reshape(-1)

        if self.bias is not None:
            delta_pred_np = delta_pred_np - self.bias

        raw_pred = raw_state + delta_pred_np
        return raw_pred


# ------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------

def build_features_and_targets(df: pd.DataFrame, feature_cols, rm_window: int = 5):
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw_all = df[feature_cols].to_numpy(dtype=np.float32)

    N = raw_all.shape[0]
    deltas_all = np.zeros_like(raw_all)
    rm_all = np.zeros_like(raw_all)

    for seq_ix, df_seq in df.groupby("seq_ix"):
        idx = df_seq.index.to_numpy()
        raw_seq = raw_all[idx]
        T = raw_seq.shape[0]

        delta_seq = np.zeros_like(raw_seq)
        if T > 1:
            delta_seq[1:] = raw_seq[1:] - raw_seq[:-1]

        rm_seq = np.zeros_like(raw_seq)
        buffer = []
        for t in range(T):
            buffer.append(delta_seq[t])
            if len(buffer) > rm_window:
                buffer = buffer[-rm_window:]
            stacked = np.stack(buffer, axis=0)
            rm_seq[t] = stacked.mean(axis=0)

        deltas_all[idx] = delta_seq
        rm_all[idx] = rm_seq

    features_all = np.concatenate([raw_all, deltas_all, rm_all], axis=1)
    return features_all, deltas_all, raw_all


def make_sequences(features: np.ndarray, deltas: np.ndarray, raw: np.ndarray, seq_len: int = 32):
    X, y_delta, y_raw = [], [], []
    N = features.shape[0]
    for i in range(N - seq_len):
        X.append(features[i:i + seq_len])
        y_delta.append(deltas[i + seq_len])
        y_raw.append(raw[i + seq_len])
    return (
        np.stack(X, axis=0),
        np.stack(y_delta, axis=0),
        np.stack(y_raw, axis=0)
    )


def train_model(
    model: PredictionModel,
    train_features: np.ndarray,
    train_deltas: np.ndarray,
    train_raw: np.ndarray,
    num_epochs: int = 3,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):

    model.train()
    model.to(DEVICE)

    X, y_delta, y_raw = make_sequences(train_features, train_deltas, train_raw, seq_len=seq_len)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_delta_tensor = torch.tensor(y_delta, dtype=torch.float32)
    y_raw_tensor = torch.tensor(y_raw, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_delta_tensor, y_raw_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb_delta, yb_raw in loader:
            xb = xb.to(DEVICE)
            yb_delta = yb_delta.to(DEVICE)
            yb_raw = yb_raw.to(DEVICE)

            optimizer.zero_grad()
            delta_pred, raw_pred = model(xb)

            loss_delta = mse_loss(delta_pred, yb_delta)
            loss_raw = mae_loss(raw_pred, yb_raw)

            # 🔥 amplitude-boosting update
            loss = 0.65 * loss_delta + 0.35 * loss_raw

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}")

    # bias calculation (unchanged)
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb_delta, _ in loader:
            xb = xb.to(DEVICE)
            yb_delta = yb_delta.to(DEVICE)
            delta_pred, _ = model(xb)
            preds_all.append(delta_pred.cpu().numpy())
            trues_all.append(yb_delta.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    bias = (preds_all - trues_all).mean(axis=0)

    bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")
    np.save(bias_path, bias)
    print("Saved bias correction vector to", bias_path)


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------

if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    scorer = ScorerStepByStep(test_file)

    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    df = scorer.dataset
    feature_cols = scorer.features

    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")

    train_features, train_deltas, train_raw = build_features_and_targets(
        df,
        feature_cols,
        rm_window=5
    )
    print("Train feature shape:", train_features.shape)
    print("Train delta shape:", train_deltas.shape)
    print("Train raw shape:", train_raw.shape)

    if os.path.exists(weights_path):
        print("⚡ Found existing weights — skipping training and going straight to scoring.")
        model = PredictionModel()
    else:
        print("🚀 No weights found — training model to create them.")
        model = PredictionModel()
        train_model(
            model,
            train_features=train_features,
            train_deltas=train_deltas,
            train_raw=train_raw,
            num_epochs=3,
            lr=1e-3,
            batch_size=64,
        )
        torch.save(model.state_dict(), weights_path)
        print("💾 Saved weights to", weights_path)
        model = PredictionModel()  # reload

    model.eval()
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(len(scorer.features)):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
