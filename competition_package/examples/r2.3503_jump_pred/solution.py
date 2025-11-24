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
# Per-feature loss weights (from offline R² of a baseline model)
#
# We penalize low-R² features more by weighting with
# exp((1 - R²) / tau). That encourages the model to reduce
# MSE where it's currently worst, which tends to increase the
# mean R² across features.
# ------------------------------------------------------------

R2_OFFLINE = np.array([
    0.235520, 0.261858, 0.344712, 0.477811,
    0.283506, 0.269687, 0.486094, 0.510849,
    0.417095, 0.353502, 0.333413, 0.351982,
    0.276168, 0.477428, 0.493655, 0.412538,
    0.314041, 0.266807, 0.475186, 0.244800,
    0.403516, 0.226987, 0.411668, 0.327140,
    0.308860, 0.389759, 0.423410, 0.419822,
    0.388730, 0.358977, 0.413292, 0.391774
], dtype=np.float32)

TAU = 0.6  # controls how aggressive the weighting is

FEATURE_WEIGHTS_RAW = np.exp((1.0 - R2_OFFLINE) / TAU)
FEATURE_WEIGHTS = FEATURE_WEIGHTS_RAW / FEATURE_WEIGHTS_RAW.mean()  # mean ~ 1.0


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Raw 32, delta1 32, rolling-mean(delta1) 32 -> 96-dim input
        self.raw_dim = 32
        self.input_dim = 96      # [raw, delta1, rm(delta1)]
        self.output_dim = 32     # predict next delta1 (32 dims)

        # Increased capacity
        self.hidden_size = 96
        self.num_layers = 1
        self.max_seq_len = 32    # how many timesteps we feed in online
        self.rm_window = 5       # rolling-mean window over delta1

        # Online state
        self.current_seq_ix = None
        self.sequence_history = []   # list of 96-dim feature vectors
        self.last_state = None       # last raw state (32,)
        self.delta_buffer = []       # last few delta1's for RM

        # LSTM over 96-dim input
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # 32 independent delta heads (one per feature)
        self.delta_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(self.output_dim)]
        )

        # Aux raw head: shared 32-dim head to stabilize amplitude
        self.fc_raw = nn.Linear(self.hidden_size, self.raw_dim)

        # NEW: jump-timing head (binary classifier: will there be a big move next step?)
        self.head_jump = nn.Linear(self.hidden_size, 1)

        # Paths
        self.weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        self.bias_path = os.path.join(CURRENT_DIR, "delta_bias.npy")

        # Load weights if present (strict=False to tolerate minor arch changes)
        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}")
            state_dict = torch.load(self.weights_path, map_location=DEVICE)
            try:
                self.load_state_dict(state_dict, strict=False)
                print("Loaded state_dict with strict=False (some keys may not match).")
            except Exception as e:
                print("Could not load existing weights:", e)

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
           delta_out: (batch_size, 32)  - predicted next delta1 (via 32 heads)
           raw_out:   (batch_size, 32)  - predicted next raw (aux head)
           jump_logit:(batch_size, 1)   - logit for jump probability
        """
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))       # (batch, seq_len, hidden_size)
        last = out[:, -1, :]                  # (batch, hidden_size)

        # 32 independent delta heads → concat to (batch, 32)
        delta_list = [head(last) for head in self.delta_heads]  # list of (batch, 1)
        delta_out = torch.cat(delta_list, dim=1)                # (batch, 32)

        # Shared raw head
        raw_out = self.fc_raw(last)           # (batch, 32)

        # Jump-timing logit
        jump_logit = self.head_jump(last)     # (batch, 1)

        return delta_out, raw_out, jump_logit

    # ---------- Online helpers ----------

    def _reset_sequence_state(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.sequence_history = []
        self.last_state = None
        self.delta_buffer = []

    def _update_rolling_mean(self, new_delta1: np.ndarray) -> np.ndarray:
        """
        Maintain rolling-mean of delta1 with window rm_window.
        Uses only past/current deltas (no peeking).
        """
        self.delta_buffer.append(new_delta1)
        if len(self.delta_buffer) > self.rm_window:
            self.delta_buffer = self.delta_buffer[-self.rm_window:]

        stacked = np.stack(self.delta_buffer, axis=0)  # (k, 32)
        rm = stacked.mean(axis=0)                      # (32,)
        return rm

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        """
        Online prediction:
        - build feature = [raw, delta1, rolling-mean(delta1)]
        - accumulate last max_seq_len features
        - predict next delta1
        - return raw + predicted_delta1 (optionally bias-corrected)

        Jump head is only used during training as an auxiliary signal.
        """

        # New sequence → reset online state
        if self.current_seq_ix != dp.seq_ix:
            self._reset_sequence_state(dp.seq_ix)

        raw_state = dp.state.astype(np.float32)  # (32,)

        # Delta1 vs last raw
        if self.last_state is None:
            delta1 = np.zeros_like(raw_state, dtype=np.float32)
        else:
            delta1 = raw_state - self.last_state

        self.last_state = raw_state

        # Rolling mean of delta1
        rm = self._update_rolling_mean(delta1)    # (32,)

        # Build 96-dim input: [raw, delta1, rm(delta1)]
        feat = np.concatenate([raw_state, delta1, rm], axis=0)  # (96,)

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

        # Predict next delta1 (main) and next raw (aux)
        delta_pred, _, _ = self.forward(x)              # delta_pred: (1, 32)
        delta_pred_np = delta_pred.detach().cpu().numpy().reshape(-1)  # (32,)

        # Apply bias correction if available
        if self.bias is not None:
            delta_pred_np = delta_pred_np - self.bias

        # Convert delta1 → raw prediction
        raw_pred = raw_state + delta_pred_np  # (32,)
        return raw_pred


# ------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------

def build_features_and_targets(df: pd.DataFrame, feature_cols, rm_window: int = 5):
    """
    Build:
      - features_all: (N, 96)  = [raw, delta1, rolling-mean(delta1)]
      - deltas1_all:  (N, 32)  = delta1
      - raw_all:      (N, 32)  = raw values
      - jump_labels:  (N,)     = binary labels for 'jump' at this step

    Jump label is computed from the norm of delta1 using an 80th percentile
    threshold, i.e., top 20% of movements are treated as 'jumps'.
    Uses only past info per sequence (no future peeking for features).
    """
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw_all = df[feature_cols].to_numpy(dtype=np.float32)  # (N, 32)

    N, dim = raw_all.shape
    delta1_all = np.zeros_like(raw_all, dtype=np.float32)
    rm_all = np.zeros_like(raw_all, dtype=np.float32)

    # compute per sequence so deltas don't cross seq boundaries
    for seq_ix, df_seq in df.groupby("seq_ix"):
        idx = df_seq.index.to_numpy()
        raw_seq = raw_all[idx]  # (T, 32)
        T = raw_seq.shape[0]

        d1 = np.zeros_like(raw_seq, dtype=np.float32)
        if T > 1:
            d1[1:] = raw_seq[1:] - raw_seq[:-1]        # delta1[t] = x[t] - x[t-1]

        # rolling mean over delta1 (cumulative, then window rm_window)
        rm_seq = np.zeros_like(raw_seq, dtype=np.float32)
        buffer = []
        for t in range(T):
            buffer.append(d1[t])
            if len(buffer) > rm_window:
                buffer = buffer[-rm_window:]
            stacked = np.stack(buffer, axis=0)  # (k, 32)
            rm_seq[t] = stacked.mean(axis=0)

        delta1_all[idx] = d1
        rm_all[idx] = rm_seq

    # build jump labels from delta1 magnitude
    # use L2 norm per timestep, then threshold at 80th percentile
    mag = np.linalg.norm(delta1_all, axis=1)   # (N,)
    thresh = np.quantile(mag, 0.8)
    jump_labels = (mag > thresh).astype(np.float32)     # (N,)

    # concat to 96-dim features
    features_all = np.concatenate([raw_all, delta1_all, rm_all], axis=1)  # (N, 96)
    return features_all, delta1_all, raw_all, jump_labels


def make_sequences(
    features: np.ndarray,
    deltas1: np.ndarray,
    raw: np.ndarray,
    jumps: np.ndarray,
    seq_len: int = 32
):
    """
    Sliding windows over features with targets:

    features: (N, 96)
    deltas1:  (N, 32)   (delta1 at each timestep)
    raw:      (N, 32)   (raw at each timestep)
    jumps:    (N,)      (jump label at each timestep)

    returns:
      X:        (num_samples, seq_len, 96)
      y_delta:  (num_samples, 32) = delta1 at t+seq_len
      y_raw:    (num_samples, 32) = raw at t+seq_len
      y_jump:   (num_samples,)    = jump label at t+seq_len
    """
    X, y_delta, y_raw, y_jump = [], [], [], []
    N = features.shape[0]
    for i in range(N - seq_len):
        X.append(features[i:i + seq_len])
        y_delta.append(deltas1[i + seq_len])
        y_raw.append(raw[i + seq_len])
        y_jump.append(jumps[i + seq_len])
    X = np.stack(X, axis=0)
    y_delta = np.stack(y_delta, axis=0)
    y_raw = np.stack(y_raw, axis=0)
    y_jump = np.array(y_jump, dtype=np.float32)
    return X, y_delta, y_raw, y_jump


def train_model(
    model: PredictionModel,
    train_features: np.ndarray,
    train_deltas1: np.ndarray,
    train_raw: np.ndarray,
    train_jumps: np.ndarray,
    num_epochs: int = 4,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    """
    Train the model on [raw, delta1, rm(delta1)] windows to predict:
      - next delta1  (main head, via 32 independent heads)
      - next raw     (aux head, helps amplitude)
      - jump timing  (aux binary classifier head)

    Main objective is per-feature weighted MSE on delta1, with auxiliary
    losses on raw level (MAE) and jump timing (BCE).
    """
    model.train()
    model.to(DEVICE)

    X, y_delta, y_raw, y_jump = make_sequences(
        train_features, train_deltas1, train_raw, train_jumps, seq_len=seq_len
    )
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_delta_tensor = torch.tensor(y_delta, dtype=torch.float32)
    y_raw_tensor = torch.tensor(y_raw, dtype=torch.float32)
    y_jump_tensor = torch.tensor(y_jump, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(
        X_tensor, y_delta_tensor, y_raw_tensor, y_jump_tensor
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mae_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    # Per-feature weights (torch tensor, normalized around mean=1)
    w = torch.tensor(FEATURE_WEIGHTS, dtype=torch.float32, device=DEVICE)
    if w.mean() != 0:
        w = w / w.mean()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb_delta, yb_raw, yb_jump in loader:
            xb = xb.to(DEVICE)
            yb_delta = yb_delta.to(DEVICE)
            yb_raw = yb_raw.to(DEVICE)
            yb_jump = yb_jump.to(DEVICE)

            optimizer.zero_grad()
            delta_pred, raw_pred, jump_logit = model(xb)   # delta: (B,32), raw: (B,32), jump_logit: (B,1)

            # Main loss: delta1 (movement) – per-feature weighted MSE
            err = (delta_pred - yb_delta) ** 2          # (batch, 32)
            weighted_err = err * w                     # broadcast weights across batch
            loss_delta = weighted_err.mean()

            # Aux loss: raw level (amplitude)
            loss_raw = mae_loss(raw_pred, yb_raw)

            # Aux loss: jump timing (BCE on logits vs labels)
            jump_logit_flat = jump_logit.view(-1)       # (batch,)
            loss_jump = bce_loss(jump_logit_flat, yb_jump)

            # Combine losses
            loss = 0.8 * loss_delta + 0.1 * loss_raw + 0.1 * loss_jump
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}")

    # ---------- Bias correction in delta1 space ----------
    model.eval()
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for xb, yb_delta, _, _ in loader:
            xb = xb.to(DEVICE)
            yb_delta = yb_delta.to(DEVICE)
            delta_pred, _, _ = model(xb)
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

    # Build features, delta1, raw, and jump labels for training
    train_features, train_deltas1, train_raw, train_jumps = build_features_and_targets(
        df,
        feature_cols,
        rm_window=5
    )
    print("Train feature shape:", train_features.shape)
    print("Train delta1 shape:", train_deltas1.shape)
    print("Train raw shape:", train_raw.shape)
    print("Train jump label shape:", train_jumps.shape)

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
            train_deltas1=train_deltas1,
            train_raw=train_raw,
            train_jumps=train_jumps,
            num_epochs=4,
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
    print("=" * 60)
