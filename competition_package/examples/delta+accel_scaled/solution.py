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

        # Raw has 32 dims, delta 32, acceleration 32 -> 96-dim input
        self.raw_dim = 32
        self.input_dim = 96     # [raw, delta_scaled, accel_scaled]
        self.output_dim = 32    # predict next delta (32 dims)

        self.hidden_size = 64
        self.num_layers = 1
        self.max_seq_len = 32   # how many timesteps we feed in online

        # Online state
        self.current_seq_ix = None
        self.sequence_history = []   # list of 96-dim feature vectors
        self.last_state = None       # last raw state (32,)
        self.last_delta = None       # last delta (32,)
        self.prev_delta_pred = None  # for momentum smoothing of predictions

        # LSTM takes 96-dim input
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Fully-connected layer that maps hidden state → delta prediction
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

        # Paths
        self.weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        self.bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")
        self.scale_delta_path = os.path.join(CURRENT_DIR, "scale_delta.npy")
        self.scale_accel_path = os.path.join(CURRENT_DIR, "scale_accel.npy")

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

        # Load scaling vectors if present
        if os.path.exists(self.scale_delta_path):
            print(f"Loading scale_delta from {self.scale_delta_path}")
            self.scale_delta = np.load(self.scale_delta_path).astype(np.float32)
        else:
            self.scale_delta = None

        if os.path.exists(self.scale_accel_path):
            print(f"Loading scale_accel from {self.scale_accel_path}")
            self.scale_accel = np.load(self.scale_accel_path).astype(np.float32)
        else:
            self.scale_accel = None

        self.to(DEVICE)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim=96)
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
        self.last_delta = None
        self.prev_delta_pred = None   # reset stabilizer state too

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        """
        Online prediction:
        - build feature = [raw, delta_scaled, accel_scaled]
        - accumulate last max_seq_len features
        - predict next delta
        - apply online stabilizers (bias, smoothing, clamp)
        - return raw + predicted_delta

        This returns raw predictions, which is what the scorer uses.
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

        # Acceleration = change in delta
        if self.last_delta is None:
            accel = np.zeros_like(delta, dtype=np.float32)
        else:
            accel = delta - self.last_delta

        self.last_state = raw_state
        self.last_delta = delta

        # Scale delta and accel using training-time std
        if self.scale_delta is not None:
            delta_scaled = delta / (self.scale_delta + 1e-6)
        else:
            delta_scaled = delta

        if self.scale_accel is not None:
            accel_scaled = accel / (self.scale_accel + 1e-6)
        else:
            accel_scaled = accel

        # Build 96-dim input: [raw, delta_scaled, accel_scaled]
        feat = np.concatenate([raw_state, delta_scaled, accel_scaled], axis=0)  # (96,)

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

        # Predict next delta
        delta_pred = self.forward(x)  # (1, 32)
        delta_pred_np = delta_pred.detach().cpu().numpy().reshape(-1)  # (32,)

        # --------- STABILIZERS (online only, no retrain needed) ---------

        # 1) Bias correction in delta space
        if self.bias is not None:
            # optionally use partial bias; 1.0 = full, 0.5 = partial
            bias_factor = 0.7
            delta_pred_np = delta_pred_np - bias_factor * self.bias

        # 2) Momentum smoothing on predicted delta (EMA over predictions)
        if self.prev_delta_pred is not None:
            # beta closer to 1 → smoother but slower to react
            beta = 0.6
            delta_pred_np = (
                beta * self.prev_delta_pred + (1.0 - beta) * delta_pred_np
            )

        # 3) Amplitude clamp based on training-time delta std
        if self.scale_delta is not None:
            # allow up to k * std for each feature
            k = 3.0
            max_step = k * self.scale_delta
            delta_pred_np = np.clip(delta_pred_np, -max_step, max_step)

        # store for next step smoothing
        self.prev_delta_pred = delta_pred_np.copy()

        # -----------------------------------------------------

        # Convert delta → raw prediction (this is what scorer sees)
        raw_pred = raw_state + delta_pred_np  # (32,)
        return raw_pred


# ------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------

def build_features_and_targets(df: pd.DataFrame, feature_cols):
    """
    Build:
      - features_all: (N, 96) = [raw, delta_scaled, accel_scaled]
      - deltas_all:   (N, 32) = delta (target)
      - seq_ids:      (N,)    = sequence ids aligned with rows
    Uses only past info per sequence (no future peeking).
    Also computes and saves per-feature std for delta and accel.
    """
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw_all = df[feature_cols].to_numpy(dtype=np.float32)  # (N, 32)

    N, dim = raw_all.shape
    deltas_all = np.zeros_like(raw_all, dtype=np.float32)
    accel_all = np.zeros_like(raw_all, dtype=np.float32)

    # compute per sequence so deltas/accel don't cross seq boundaries
    for seq_ix, df_seq in df.groupby("seq_ix"):
        idx = df_seq.index.to_numpy()
        raw_seq = raw_all[idx]  # (T, 32)
        T = raw_seq.shape[0]

        delta_seq = np.zeros_like(raw_seq, dtype=np.float32)
        if T > 1:
            delta_seq[1:] = raw_seq[1:] - raw_seq[:-1]

        accel_seq = np.zeros_like(raw_seq, dtype=np.float32)
        if T > 2:
            accel_seq[1:] = delta_seq[1:] - delta_seq[:-1]

        deltas_all[idx] = delta_seq
        accel_all[idx] = accel_seq

    # compute std of delta and accel for scaling (over all sequences)
    delta_std = deltas_all.std(axis=0)
    accel_std = accel_all.std(axis=0)

    # avoid division by zero
    delta_std = np.clip(delta_std, 1e-6, None)
    accel_std = np.clip(accel_std, 1e-6, None)

    # save scaling vectors
    scale_delta_path = os.path.join(CURRENT_DIR, "scale_delta.npy")
    scale_accel_path = os.path.join(CURRENT_DIR, "scale_accel.npy")
    np.save(scale_delta_path, delta_std.astype(np.float32))
    np.save(scale_accel_path, accel_std.astype(np.float32))
    print("Saved scale_delta to", scale_delta_path)
    print("Saved scale_accel to", scale_accel_path)

    # concat to 96-dim features: [raw, delta_scaled, accel_scaled]
    delta_scaled = deltas_all / delta_std
    accel_scaled = accel_all / accel_std
    features_all = np.concatenate([raw_all, delta_scaled, accel_scaled], axis=1)  # (N, 96)

    # seq_ids aligned with rows
    seq_ids = df["seq_ix"].to_numpy(dtype=np.int32)

    return features_all, deltas_all, seq_ids


def make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int = 32):
    """
    Sliding windows over features with delta targets.

    features: (N, 96)
    targets:  (N, 32)   (deltas)
    returns:
      X: (num_samples, seq_len, 96)
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
    val_features: np.ndarray = None,
    val_deltas: np.ndarray = None,
    num_epochs: int = 3,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    """
    Train the model on [raw, delta_scaled, accel_scaled] windows to predict next delta.
    Also computes and saves per-feature bias on delta predictions.
    Uses a validation set if provided and prints validation R² (delta space).
    """
    model.train()
    model.to(DEVICE)

    # ----- training dataset -----
    X_train, y_train = make_sequences(train_features, train_deltas, seq_len=seq_len)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # ----- validation dataset (optional) -----
    if val_features is not None and val_deltas is not None:
        X_val, y_val = make_sequences(val_features, val_deltas, seq_len=seq_len)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        val_loader = None
        val_dataset = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        # ----- train loop -----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_dataset)

        # ----- validation loop -----
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_dataset)
            print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.6f}")

    # ----------------- compute validation R² (delta space) -----------------
    if val_loader is not None:
        model.eval()
        val_preds_all = []
        val_trues_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                p = model(xb).cpu().numpy()
                t = yb.cpu().numpy()
                val_preds_all.append(p)
                val_trues_all.append(t)
        val_preds_all = np.concatenate(val_preds_all, axis=0)  # (Nv, 32)
        val_trues_all = np.concatenate(val_trues_all, axis=0)  # (Nv, 32)

        t_mean = val_trues_all.mean(axis=0, keepdims=True)
        ss_res = ((val_trues_all - val_preds_all) ** 2).sum(axis=0)
        ss_tot = ((val_trues_all - t_mean) ** 2).sum(axis=0) + 1e-9
        r2_per_dim = 1.0 - ss_res / ss_tot
        mean_r2 = float(r2_per_dim.mean())

        print("\nValidation R² per feature (delta space):")
        print(r2_per_dim)
        print(f"Validation mean R² (delta space): {mean_r2:.6f}\n")

    # ----------------- compute & save TRAIN bias (delta space) -------------
    model.eval()
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for xb, yb in train_loader:
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
    np.save(bias_path, bias.astype(np.float32))
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

    # Build features and deltas for ALL sequences (also saves scaling vectors)
    features_all, deltas_all, seq_ids = build_features_and_targets(df, feature_cols)
    print("All feature shape:", features_all.shape)
    print("All delta shape:", deltas_all.shape)

    # ------------- sequence-wise train/validation split -------------
    unique_seqs = np.unique(seq_ids)
    num_seqs = len(unique_seqs)
    cut = int(0.8 * num_seqs)   # first 80% for training, last 20% for validation

    train_seq_ids = unique_seqs[:cut]
    val_seq_ids = unique_seqs[cut:]

    train_mask = np.isin(seq_ids, train_seq_ids)
    val_mask = np.isin(seq_ids, val_seq_ids)

    train_features = features_all[train_mask]
    train_deltas = deltas_all[train_mask]

    val_features = features_all[val_mask]
    val_deltas = deltas_all[val_mask]

    print(f"Total sequences: {num_seqs}")
    print(f"Train sequences: {len(train_seq_ids)}  ({len(train_features)} rows)")
    print(f"Val   sequences: {len(val_seq_ids)}  ({len(val_features)} rows)")

    # If weights exist → skip training, just score
    if os.path.exists(weights_path):
        print("⚡ Found existing weights — skipping training and going straight to scoring.")
        model = PredictionModel()  # will load weights + bias + scaling in __init__
    else:
        print("🚀 No weights found — training model to create them.")
        model = PredictionModel()  # random init
        train_model(
            model,
            train_features=train_features,
            train_deltas=train_deltas,
            val_features=val_features,
            val_deltas=val_deltas,
            num_epochs=3,
            lr=1e-3,
            batch_size=64,
        )
        torch.save(model.state_dict(), weights_path)
        print("💾 Saved weights to", weights_path)
        # Reload to ensure we're using clean loaded state + bias + scaling
        model = PredictionModel()

    # Score using online interface (this uses raw predictions from predict())
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
