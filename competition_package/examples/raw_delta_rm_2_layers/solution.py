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

        # Raw has 32 dims, delta 32, rolling-mean 32 -> 96-dim input
        self.raw_dim = 32
        self.input_dim = 96     # [raw, delta, rolling mean]
        self.output_dim = 32    # predict next delta (32 dims)

        # Slightly larger LSTM with 2 layers for better dynamics,
        # but still small enough to stay under 1h with 3 epochs on a 4060Ti.
        self.hidden_size = 96
        self.num_layers = 2
        self.max_seq_len = 32   # how many timesteps we feed in online
        self.rm_window = 5      # rolling-mean window over deltas

        # Online state
        self.current_seq_ix = None
        self.sequence_history = []   # list of 96-dim feature vectors
        self.last_state = None       # last raw state (32,)
        self.delta_buffer = []       # last few deltas for RM

        # LSTM takes 96-dim input
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.15 if self.num_layers > 1 else 0.0
        )

        # Two heads:
        #  - fc_delta: predicts next delta (for main forecasting + bias)
        #  - fc_raw:   predicts next raw (auxiliary to fix amplitude)
        self.fc_delta = nn.Linear(self.hidden_size, self.output_dim)
        self.fc_raw = nn.Linear(self.hidden_size, self.raw_dim)

        # Paths
        self.weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        self.bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")

        # Load weights if present
        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}")
            state_dict = torch.load(self.weights_path, map_location=DEVICE)
            self.load_state_dict(state_dict)

        # Load bias if present
        if os.path.exists(self.bias_path):
            print(f"Loading bias from {self.bias_path}")
            self.bias = np.load(self.bias_path).astype(np.float32)
        else:
            self.bias = None

        self.to(DEVICE)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim=96)
        returns:
           delta_out: (batch_size, 32)  - predicted next delta
           raw_out:   (batch_size, 32)  - predicted next raw (aux head)
        """
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))       # (batch, seq_len, hidden_size)
        last = out[:, -1, :]                  # (batch, hidden_size)

        delta_out = self.fc_delta(last)       # (batch, 32)
        raw_out = self.fc_raw(last)           # (batch, 32)
        return delta_out, raw_out

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
        - build feature = [raw, delta, rolling-mean(delta)]
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

        # Build 96-dim input: [raw, delta, rm]
        feat = np.concatenate([raw_state, delta, rm], axis=0)  # (96,)

        # Append to history
        self.sequence_history.append(feat)
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        # If scorer doesn't need prediction yet, return None
        if not dp.need_prediction:
            return None

        # Build tensor (1, seq_len, 96)
        seq = np.stack(self.sequence_history, axis=0)
        x = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Predict next delta (main) and next raw (aux)
        delta_pred, _ = self.forward(x)              # each: (1, 32)
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
      - features_all: (N, 96) = [raw, delta, rolling-mean(delta)]
      - deltas_all:   (N, 32) = delta
      - raw_all:      (N, 32) = raw values
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

        # rolling mean over deltas (cumulative, then window rm_window)
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

    # concat to 96-dim features
    features_all = np.concatenate([raw_all, deltas_all, rm_all], axis=1)  # (N, 96)
    return features_all, deltas_all, raw_all


def make_sequences(features: np.ndarray, deltas: np.ndarray, raw: np.ndarray, seq_len: int = 32):
    """
    Sliding windows over features with targets:

    features: (N, 96)
    deltas:   (N, 32)   (delta at each timestep)
    raw:      (N, 32)   (raw at each timestep)

    returns:
      X:        (num_samples, seq_len, 96)
      y_delta:  (num_samples, 32) = delta at t+seq_len
      y_raw:    (num_samples, 32) = raw at t+seq_len
    """
    X, y_delta, y_raw = [], [], []
    N = features.shape[0]
    for i in range(N - seq_len):
        X.append(features[i:i + seq_len])
        y_delta.append(deltas[i + seq_len])
        y_raw.append(raw[i + seq_len])
    X = np.stack(X, axis=0)
    y_delta = np.stack(y_delta, axis=0)
    y_raw = np.stack(y_raw, axis=0)
    return X, y_delta, y_raw


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
    """
    Train the model on [raw, delta, rm] windows to predict:
      - next delta (main head)
      - next raw   (aux head, helps amplitude)
    """
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

            # Main loss: delta (movement)
            loss_delta = mse_loss(delta_pred, yb_delta)
            # Aux loss: raw level (amplitude)
            loss_raw = mae_loss(raw_pred, yb_raw)

            loss = 0.85 * loss_delta + 0.15 * loss_raw
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}")

    # compute and save bias in delta space (using main head only)
    model.eval()
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for xb, yb_delta, _ in loader:
            xb = xb.to(DEVICE)
            yb_delta = yb_delta.to(DEVICE)
            delta_pred, _ = model(xb)
            p = delta_pred.cpu().numpy()
            t = yb_delta.cpu().numpy()
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

    # Build features, deltas, and raw for training
    train_features, train_deltas, train_raw = build_features_and_targets(
        df,
        feature_cols,
        rm_window=5
    )
    print("Train feature shape:", train_features.shape)
    print("Train delta shape:", train_deltas.shape)
    print("Train raw shape:", train_raw.shape)

    # If weights exist → skip training, just score
    if os.path.exists(weights_path):
        print("⚡ Found existing weights — skipping training and going straight to scoring.")
        model = PredictionModel()  # will load weights + bias in __init__
    else:
        print("🚀 No weights found — training model to create them.")
        model = PredictionModel()  # random init
        train_model(
            model,
            train_features=train_features,
            train_deltas=train_deltas,
            train_raw=train_raw,
            num_epochs=3,     # you can drop to 2 if runtime is tight
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
