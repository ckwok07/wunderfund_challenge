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

        # Raw has 32 dims.
        # We build features = [raw, delta, ema_delta, vol_raw, momentum]
        #   raw        : 32
        #   delta      : 32
        #   ema_delta  : 32
        #   vol_raw    : 32
        #   momentum   : 32
        # => 160-dim input
        self.raw_dim = 32
        self.input_dim = 32 * 5   # 160
        self.output_dim = 32      # predict next delta (32 dims)

        # LSTM config
        self.hidden_size = 96
        self.num_layers = 2
        self.max_seq_len = 32     # how many timesteps we feed in online

        # Feature-engineering hyperparams (must match offline builder)
        self.ema_alpha = 0.3      # EMA smoothing for delta
        self.vol_window = 15      # rolling std window over raw

        # Online state
        self.current_seq_ix = None
        self.sequence_history = []   # list of 160-dim feature vectors
        self.last_state = None       # last raw state (32,)
        self.last_ema_delta = None   # last EMA(delta) (32,)
        self.raw_buffer = []         # for rolling std of raw

        # LSTM takes 160-dim input
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.15 if self.num_layers > 1 else 0.0
        )

        # Two heads:
        #  - fc_delta: predicts next delta (for main forecasting + bias)
        #  - fc_raw:   predicts next raw (auxiliary, helps amplitude)
        self.fc_delta = nn.Linear(self.hidden_size, self.output_dim)
        self.fc_raw = nn.Linear(self.hidden_size, self.raw_dim)

        # Paths
        # Use a new filename so we don't clash with older models
        self.weights_path = os.path.join(CURRENT_DIR, "lstm_weights_features_v2.pt")
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
        x: (batch_size, seq_len, input_dim=160)
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
        self.last_ema_delta = None
        self.raw_buffer = []

    def _update_online_features(
        self,
        raw_state: np.ndarray
    ):
        """
        Given current raw_state (32,),
        compute delta, ema_delta, vol_raw, momentum
        using only past/current info. Used in online predict().
        """
        # Delta vs last raw
        if self.last_state is None:
            delta = np.zeros_like(raw_state, dtype=np.float32)
        else:
            delta = raw_state - self.last_state

        self.last_state = raw_state

        # EMA(delta)
        if self.last_ema_delta is None:
            ema_delta = np.zeros_like(delta, dtype=np.float32)
        else:
            ema_delta = self.ema_alpha * delta + (1.0 - self.ema_alpha) * self.last_ema_delta

        self.last_ema_delta = ema_delta

        # Rolling std of raw (volatility)
        self.raw_buffer.append(raw_state.copy())
        if len(self.raw_buffer) > self.vol_window:
            self.raw_buffer = self.raw_buffer[-self.vol_window:]
        stacked_raw = np.stack(self.raw_buffer, axis=0)  # (k, 32)
        vol_raw = stacked_raw.std(axis=0).astype(np.float32)

        # Momentum: delta - ema_delta
        momentum = delta - ema_delta

        return delta, ema_delta, vol_raw, momentum

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        """
        Online prediction:
        - build feature = [raw, delta, ema_delta, vol_raw, momentum]
        - accumulate last max_seq_len features
        - predict next delta
        - return raw + predicted_delta (optionally bias-corrected)
        """

        # New sequence → reset online state
        if self.current_seq_ix != dp.seq_ix:
            self._reset_sequence_state(dp.seq_ix)

        raw_state = dp.state.astype(np.float32)  # (32,)

        # Build delta / ema_delta / vol_raw / momentum using only past/current info
        delta, ema_delta, vol_raw, momentum = self._update_online_features(raw_state)

        # Build 160-dim input: [raw, delta, ema_delta, vol_raw, momentum]
        feat = np.concatenate(
            [raw_state, delta, ema_delta, vol_raw, momentum],
            axis=0
        ).astype(np.float32)  # (160,)

        # Append to history
        self.sequence_history.append(feat)
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        # If scorer doesn't need prediction yet, return None
        if not dp.need_prediction:
            return None

        # Build tensor (1, seq_len, 160)
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

def build_features_and_targets(
    df: pd.DataFrame,
    feature_cols,
    ema_alpha: float = 0.3,
    vol_window: int = 15
):
    """
    Build:
      - features_all: (N, 160) = [raw, delta, ema_delta, vol_raw, momentum]
      - deltas_all:   (N, 32)  = delta
      - raw_all:      (N, 32)  = raw values

    Uses only past info per sequence (no future peeking).
    raw is assumed already standardized by the competition.
    """
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw_all = df[feature_cols].to_numpy(dtype=np.float32)  # (N, 32)

    N, dim = raw_all.shape
    deltas_all = np.zeros_like(raw_all, dtype=np.float32)
    ema_all = np.zeros_like(raw_all, dtype=np.float32)
    vol_all = np.zeros_like(raw_all, dtype=np.float32)
    mom_all = np.zeros_like(raw_all, dtype=np.float32)

    # compute per sequence so nothing crosses seq boundaries
    for seq_ix, df_seq in df.groupby("seq_ix"):
        idx = df_seq.index.to_numpy()
        raw_seq = raw_all[idx]  # (T, 32)
        T = raw_seq.shape[0]

        # Per-sequence state
        last_raw = None
        ema_delta = np.zeros(dim, dtype=np.float32)
        raw_buffer = []

        for t in range(T):
            raw_t = raw_seq[t]

            # delta
            if last_raw is None:
                delta_t = np.zeros(dim, dtype=np.float32)
            else:
                delta_t = raw_t - last_raw
            last_raw = raw_t

            # EMA(delta)
            ema_delta = ema_alpha * delta_t + (1.0 - ema_alpha) * ema_delta

            # rolling std on raw
            raw_buffer.append(raw_t)
            if len(raw_buffer) > vol_window:
                raw_buffer = raw_buffer[-vol_window:]
            stacked_raw = np.stack(raw_buffer, axis=0)  # (k, 32)
            vol_t = stacked_raw.std(axis=0).astype(np.float32)

            # momentum = delta - ema_delta
            mom_t = delta_t - ema_delta

            # write out
            deltas_all[idx[t]] = delta_t
            ema_all[idx[t]] = ema_delta
            vol_all[idx[t]] = vol_t
            mom_all[idx[t]] = mom_t

    # concat to 160-dim features:
    # [raw, delta, ema_delta, vol_raw, momentum]
    features_all = np.concatenate(
        [raw_all, deltas_all, ema_all, vol_all, mom_all],
        axis=1
    )  # (N, 160)

    return features_all, deltas_all, raw_all


def make_sequences(
    features: np.ndarray,
    deltas: np.ndarray,
    raw: np.ndarray,
    seq_len: int = 32
):
    """
    Sliding windows over features with targets:

    features: (N, 160)
    deltas:   (N, 32)   (delta at each timestep)
    raw:      (N, 32)   (raw at each timestep)

    returns:
      X:        (num_samples, seq_len, 160)
      y_delta:  (num_samples, 32) = delta at t+seq_len
      y_raw:    (num_samples, 32) = raw   at t+seq_len
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
    Train the model on [raw, delta, ema_delta, vol_raw, momentum] windows to predict:
      - next delta (main head)
      - next raw   (aux head, helps amplitude)
    """
    model.train()
    model.to(DEVICE)

    X, y_delta, y_raw = make_sequences(
        train_features,
        train_deltas,
        train_raw,
        seq_len=seq_len
    )
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

    # Where to store weights (v2 name to avoid shape mismatch with old models)
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights_features_v2.pt")

    # Build features, deltas, and raw for training
    train_features, train_deltas, train_raw = build_features_and_targets(
        df,
        feature_cols,
        ema_alpha=0.3,
        vol_window=15
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
