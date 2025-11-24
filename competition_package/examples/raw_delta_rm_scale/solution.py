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

        # Raw 32, delta 32, rolling-mean 32, per-feature amplitude 32 -> 128-dim input
        self.raw_dim = 32
        self.input_dim = 128    # [raw, delta, rolling mean, |delta|]
        self.output_dim = 32    # predict next delta (32 dims)

        self.hidden_size = 64
        self.num_layers = 1
        self.max_seq_len = 32   # how many timesteps we feed in online
        self.rm_window = 5      # rolling-mean window over deltas

        # Online state
        self.current_seq_ix = None
        self.sequence_history = []   # list of 128-dim feature vectors
        self.last_state = None       # last raw state (32,)
        self.delta_buffer = []       # last few deltas for RM

        # LSTM takes 128-dim input
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # FC maps to 32-dim delta prediction
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

        # Paths
        self.weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        self.bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")

        # Load weights if present (guard against shape mismatch after architecture change)
        if os.path.exists(self.weights_path):
            try:
                print(f"Loading weights from {self.weights_path}")
                state_dict = torch.load(self.weights_path, map_location=DEVICE)
                self.load_state_dict(state_dict)
            except Exception as e:
                print(
                    f"⚠️ Could not load existing weights (likely shape mismatch "
                    f"after architecture change). Starting from scratch. Error: {e}"
                )

        # Load bias if present
        if os.path.exists(self.bias_path):
            print(f"Loading bias from {self.bias_path}")
            self.bias = np.load(self.bias_path).astype(np.float32)
        else:
            self.bias = None

        self.to(DEVICE)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim=128)
        returns: (batch_size, 32) predicted delta
        """
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))       # (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])          # (batch, 32)
        return out

    def _reset_sequence_state(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.sequence_history = []
        self.last_state = None
        self.delta_buffer = []

    def _update_rolling_mean(self, new_delta: np.ndarray) -> np.ndarray:
        """
        Maintain rolling-mean of deltas with window rm_window.
        Uses only past/current deltas (no peeking).
        """
        self.delta_buffer.append(new_delta)
        if len(self.delta_buffer) > self.rm_window:
            self.delta_buffer = self.delta_buffer[-self.rm_window:]

        # cumulative for first few steps, then fixed window
        stacked = np.stack(self.delta_buffer, axis=0)  # (k, 32)
        rm = stacked.mean(axis=0)                      # (32,)
        return rm

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        """
        Online prediction:
        - build feature = [raw, delta, rolling-mean(delta), |delta|]
        - accumulate last max_seq_len features
        - predict next delta
        - return raw + predicted_delta (optionally bias-corrected)
        """

        # New sequence → reset online state
        if self.current_seq_ix != dp.seq_ix:
            self._reset_sequence_state(dp.seq_ix)

        raw_state = dp.state.astype(np.float32)  # (32,)

        # Delta vs last raw
        if self.last_state is None:
            delta = np.zeros_like(raw_state, dtype=np.float32)
        else:
            delta = raw_state - self.last_state

        self.last_state = raw_state

        # Rolling mean of deltas
        rm = self._update_rolling_mean(delta)    # (32,)

        # Per-feature amplitude: absolute delta at this step
        amp = np.abs(delta).astype(np.float32)   # (32,)

        # Build 128-dim input: [raw, delta, rm, |delta|]
        feat = np.concatenate([raw_state, delta, rm, amp], axis=0)  # (128,)

        # Append to history
        self.sequence_history.append(feat)
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        # If scorer doesn't need prediction yet, return None
        if not dp.need_prediction:
            return None

        # Build tensor (1, seq_len, 128)
        seq = np.stack(self.sequence_history, axis=0)
        x = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Predict next delta
        delta_pred = self.forward(x)  # (1, 32)
        delta_pred_np = delta_pred.detach().cpu().numpy().reshape(-1)  # (32,)

        # Apply bias correction if available
        if self.bias is not None:
            delta_pred_np = delta_pred_np - self.bias

        # Convert delta → raw prediction
        raw_pred = raw_state + delta_pred_np  # (32,)
        return raw_pred


# ------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------

def build_features_and_targets(df: pd.DataFrame, feature_cols, rm_window: int = 5):
    """
    Build:
      - features_all: (N, 128) = [raw, delta, rolling-mean(delta), |delta|]
      - deltas_all:   (N, 32)  = delta
    Uses only past info per sequence (no future peeking).
    """
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw_all = df[feature_cols].to_numpy(dtype=np.float32)  # (N, 32)

    N, dim = raw_all.shape
    deltas_all = np.zeros_like(raw_all, dtype=np.float32)
    rm_all = np.zeros_like(raw_all, dtype=np.float32)

    # compute per sequence so deltas don't cross seq boundaries
    for seq_ix, df_seq in df.groupby("seq_ix"):
        idx = df_seq.index.to_numpy()
        raw_seq = raw_all[idx]  # (T, 32)
        T = raw_seq.shape[0]

        delta_seq = np.zeros_like(raw_seq, dtype=np.float32)
        if T > 1:
            delta_seq[1:] = raw_seq[1:] - raw_seq[:-1]

        # rolling mean over deltas (cumulative, then fixed window rm_window)
        rm_seq = np.zeros_like(raw_seq, dtype=np.float32)
        buffer = []
        for t in range(T):
            buffer.append(delta_seq[t])
            if len(buffer) > rm_window:
                buffer = buffer[-rm_window:]
            stacked = np.stack(buffer, axis=0)  # (k, 32)
            rm_seq[t] = stacked.mean(axis=0)

        deltas_all[idx] = delta_seq
        rm_all[idx] = rm_seq

    # per-feature amplitude = |delta|
    amp_all = np.abs(deltas_all).astype(np.float32)  # (N, 32)

    # concat to 128-dim features
    features_all = np.concatenate([raw_all, deltas_all, rm_all, amp_all], axis=1)  # (N, 128)
    return features_all, deltas_all


def make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int = 32):
    """
    Sliding windows over features with delta targets.

    features: (N, input_dim)  (here input_dim = 128)
    targets:  (N, 32)         (deltas)
    returns:
      X: (num_samples, seq_len, input_dim)
      y: (num_samples, 32)      = delta at time t+seq_len
    """
    X, y = [], []
    N = features.shape[0]
    for i in range(N - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    return X, y


def train_model(
    model: PredictionModel,
    train_features: np.ndarray,
    train_deltas: np.ndarray,
    num_epochs: int = 3,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    """
    Train the model on [raw, delta, rm, |delta|] windows to predict next delta.
    """
    model.train()
    model.to(DEVICE)

    X, y = make_sequences(train_features, train_deltas, seq_len=seq_len)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}")

    # compute and save bias in delta space
    model.eval()
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            p = model(xb).cpu().numpy()
            t = yb.cpu().numpy()
            preds_all.append(p)
            trues_all.append(t)
    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    bias = (preds_all - trues_all).mean(axis=0)  # (32,)

    bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")
    np.save(bias_path, bias)
    print("Saved bias correction vector to", bias_path)


# ------------------------------------------------------------
# Main script: train once, then just score
# ------------------------------------------------------------

if __name__ == "__main__":
    # Path to train parquet
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    df = scorer.dataset
    feature_cols = scorer.features  # 32 raw variables

    # Where to store weights
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")

    # Build features and deltas for training
    train_features, train_deltas = build_features_and_targets(df, feature_cols, rm_window=5)
    print("Train feature shape:", train_features.shape)
    print("Train delta shape:", train_deltas.shape)

    # If weights exist → skip training, just score
    if os.path.exists(weights_path):
        print("⚡ Found existing weights — skipping training and going straight to scoring.")
        model = PredictionModel()  # will try to load weights + bias in __init__
    else:
        print("🚀 No weights found — training model to create them.")
        model = PredictionModel()  # random init
        train_model(
            model,
            train_features=train_features,
            train_deltas=train_deltas,
            num_epochs=3,
            lr=1e-3,
            batch_size=64,
        )
        torch.save(model.state_dict(), weights_path)
        print("💾 Saved weights to", weights_path)
        # Reload to ensure we're using clean loaded state + bias
        model = PredictionModel()

    # Score using online interface
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
    print("Try submitting an archive with solution.py file")
    print("to test the solution submission mechanism!")
    print("=" * 60)
