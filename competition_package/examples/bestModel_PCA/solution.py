import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# ------------------------------------------------------------
# Device & imports
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", DEVICE)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")  # import scorer + datapoint
from utils import DataPoint, ScorerStepByStep


# ============================================================
# Loss weighting based on baseline R²
# ============================================================

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

TAU = 0.6
FEATURE_WEIGHTS = np.exp((1.0 - R2_OFFLINE) / TAU)
FEATURE_WEIGHTS /= FEATURE_WEIGHTS.mean()


# ============================================================
# PCA on delta1
# ============================================================

def compute_pca_delta(d1: np.ndarray, k: int, tag: str):
    """
    d1: (N, 32) float32
    k: number of principal components
    tag: string to differentiate files, e.g. 'pca4', 'pca8'
    Saves PCA params to CURRENT_DIR/f"pca_delta_{tag}.npz"
    Returns Z: (N, k) float32
    """
    assert d1.ndim == 2 and d1.shape[1] == 32
    if k <= 0:
        return np.zeros((d1.shape[0], 0), dtype=np.float32)

    # center
    mean = d1.mean(axis=0)  # (32,)
    X = (d1 - mean).astype(np.float64)

    # SVD for PCA
    # X = U S V^T, principal directions = first k rows of V^T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:k].T  # (32, k)

    Z = X @ components      # (N, k)

    pca_path = os.path.join(CURRENT_DIR, f"pca_delta_{tag}.npz")
    np.savez(
        pca_path,
        mean=mean.astype(np.float32),
        components=components.astype(np.float32)
    )
    print(f"Saved PCA (k={k}) to {pca_path}")
    return Z.astype(np.float32)


# ============================================================
# Model definition
# ============================================================

class PredictionModel(nn.Module):

    def __init__(self,
                 hidden_size: int = 256,
                 input_dim: int = 160,
                 pca_tag: str | None = None,
                 weights_path: str | None = None,
                 bias_path: str | None = None):
        super().__init__()

        self.raw_dim = 32
        self.output_dim = 32

        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_layers = 1
        self.max_seq_len = 32
        self.rm_window = 5

        # PCA params (optional)
        self.pca_tag = pca_tag
        self.pca_mean = None
        self.pca_components = None
        if pca_tag is not None:
            pca_file = os.path.join(CURRENT_DIR, f"pca_delta_{pca_tag}.npz")
            if os.path.exists(pca_file):
                print(f"Loading PCA params from {pca_file}")
                data = np.load(pca_file)
                self.pca_mean = data["mean"].astype(np.float32)          # (32,)
                self.pca_components = data["components"].astype(np.float32)  # (32, k)
            else:
                print(f"WARNING: PCA file {pca_file} not found, skipping PCA features.")
                self.pca_tag = None

        # online buffers
        self.current_seq_ix = None
        self.sequence_history = []
        self.last_state = None
        self.delta_buffer = []
        self.raw_buffer = []

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Heads
        self.delta_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(self.output_dim)]
        )
        self.fc_raw = nn.Linear(self.hidden_size, self.raw_dim)
        self.head_jump = nn.Linear(self.hidden_size, 1)

        # paths
        if weights_path is None:
            self.weights_path = os.path.join(CURRENT_DIR, "lstm_h256.pt")
        else:
            self.weights_path = weights_path

        if bias_path is None:
            self.bias_path = os.path.join(CURRENT_DIR, "delta_bias_h256.npy")
        else:
            self.bias_path = bias_path

        # load weights if present
        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}")
            state = torch.load(self.weights_path, map_location=DEVICE)
            self.load_state_dict(state, strict=False)

        # load bias if present
        if os.path.exists(self.bias_path):
            print(f"Loading bias from {self.bias_path}")
            self.bias = np.load(self.bias_path).astype(np.float32)
        else:
            self.bias = None

        self.to(DEVICE)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]
        delta_pred = torch.cat([h(last) for h in self.delta_heads], dim=1)
        raw_pred = self.fc_raw(last)
        jump_logit = self.head_jump(last)
        return delta_pred, raw_pred, jump_logit

    def _reset(self, seq_ix):
        self.current_seq_ix = seq_ix
        self.sequence_history = []
        self.last_state = None
        self.delta_buffer = []
        self.raw_buffer = []

    def _update_rm_delta(self, d1):
        self.delta_buffer.append(d1)
        if len(self.delta_buffer) > self.rm_window:
            self.delta_buffer = self.delta_buffer[-self.rm_window:]
        arr = np.stack(self.delta_buffer, axis=0)
        return arr.mean(axis=0), arr.std(axis=0)

    def _update_rm_raw(self, r):
        self.raw_buffer.append(r)
        if len(self.raw_buffer) > self.rm_window:
            self.raw_buffer = self.raw_buffer[-self.rm_window:]
        return np.stack(self.raw_buffer, axis=0).mean(axis=0)

    def _pca_features(self, d1: np.ndarray):
        """
        d1: (32,) float32
        returns: (k,) float32 if PCA is enabled, else empty array (0,)
        """
        if self.pca_tag is None or self.pca_components is None or self.pca_mean is None:
            return np.zeros((0,), dtype=np.float32)
        centered = (d1 - self.pca_mean).astype(np.float32)        # (32,)
        # (32,) @ (32,k) -> (k,)
        return centered @ self.pca_components

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        if self.current_seq_ix != dp.seq_ix:
            self._reset(dp.seq_ix)

        raw = dp.state.astype(np.float32)      # (32,)
        if self.last_state is None:
            d1 = np.zeros_like(raw)
        else:
            d1 = raw - self.last_state
        self.last_state = raw

        rm_delta, vol_delta = self._update_rm_delta(d1)
        rm_raw = self._update_rm_raw(raw)
        norm_d1 = d1 / (vol_delta + 1e-6)
        norm_d1 = np.clip(norm_d1, -8, 8)

        # PCA features on delta1
        pca_feat = self._pca_features(d1)      # (k,) or (0,)

        feat = np.concatenate([raw, d1, rm_delta, rm_raw, norm_d1, pca_feat], axis=0)
        self.sequence_history.append(feat.astype(np.float32))
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        if not dp.need_prediction:
            return None

        x = torch.tensor(
            np.stack(self.sequence_history, axis=0),
            dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        delta_pred, _, _ = self.forward(x)
        delta_pred = delta_pred.cpu().numpy().reshape(-1)

        if hasattr(self, "bias") and self.bias is not None:
            delta_pred = delta_pred - self.bias

        return raw + delta_pred


# ============================================================
# Training utilities
# ============================================================

def build_features_and_targets(df, cols, rm_window=5, pca_k: int = 0, pca_tag: str = "base"):
    """
    Build:
      - feats: (N, 160 + pca_k) = [raw, d1, rm_delta, rm_raw, norm, pca_delta_k]
      - d1:    (N, 32)
      - raw:   (N, 32)
      - jump:  (N,)
    Also:
      - fit PCA on delta1 if pca_k > 0 and save params for online usage.
    """
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw = df[cols].to_numpy(np.float32)

    N = len(df)
    d1 = np.zeros_like(raw, np.float32)
    rm_delta = np.zeros_like(raw, np.float32)
    rm_raw = np.zeros_like(raw, np.float32)
    vol = np.zeros_like(raw, np.float32)

    # per-sequence stats
    for _, df_seq in df.groupby("seq_ix"):
        idx = df_seq.index.to_numpy()
        raw_seq = raw[idx]
        T = raw_seq.shape[0]
        d = np.zeros_like(raw_seq, np.float32)
        if T > 1:
            d[1:] = raw_seq[1:] - raw_seq[:-1]

        buf_d, buf_r = [], []
        rm_delta_seq = np.zeros_like(raw_seq, np.float32)
        rm_raw_seq = np.zeros_like(raw_seq, np.float32)
        vol_seq = np.zeros_like(raw_seq, np.float32)

        for t in range(T):
            buf_d.append(d[t])
            if len(buf_d) > rm_window:
                buf_d = buf_d[-rm_window:]
            arr_d = np.stack(buf_d, 0)
            rm_delta_seq[t] = arr_d.mean(0)
            vol_seq[t] = arr_d.std(0)

            buf_r.append(raw_seq[t])
            if len(buf_r) > rm_window:
                buf_r = buf_r[-rm_window:]
            arr_r = np.stack(buf_r, 0)
            rm_raw_seq[t] = arr_r.mean(0)

        d1[idx] = d
        rm_delta[idx] = rm_delta_seq
        rm_raw[idx] = rm_raw_seq
        vol[idx] = vol_seq

    norm = d1 / (vol + 1e-6)
    norm = np.clip(norm, -8, 8)
    mag = np.linalg.norm(d1, axis=1)
    jump = (mag > np.quantile(mag, 0.8)).astype(np.float32)

    # PCA on delta1 if requested
    if pca_k > 0:
        pca_delta = compute_pca_delta(d1, k=pca_k, tag=pca_tag)   # (N, pca_k)
        feats = np.concatenate([raw, d1, rm_delta, rm_raw, norm, pca_delta], axis=1)
    else:
        feats = np.concatenate([raw, d1, rm_delta, rm_raw, norm], axis=1)

    return feats.astype(np.float32), d1.astype(np.float32), raw.astype(np.float32), jump.astype(np.float32)


def make_sequences(feat, d1, raw, jump, seq_len=32):
    X, yd, yr, yj = [], [], [], []
    N = feat.shape[0]
    for i in range(N - seq_len):
        X.append(feat[i:i + seq_len])
        yd.append(d1[i + seq_len])
        yr.append(raw[i + seq_len])
        yj.append(jump[i + seq_len])
    return (np.stack(X), np.stack(yd), np.stack(yr), np.array(yj, np.float32))


def train_model(model, feat, d1, raw, jump, epochs=4, seq_len=32, lr=1e-3, batch=64):
    X, yd, yr, yj = make_sequences(feat, d1, raw, jump, seq_len)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(yd, dtype=torch.float32),
        torch.tensor(yr, dtype=torch.float32),
        torch.tensor(yj, dtype=torch.float32),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mae = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()
    w = torch.tensor(FEATURE_WEIGHTS, device=DEVICE)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb_d1, yb_raw, yb_jump in dl:
            xb, yb_d1, yb_raw, yb_jump = (
                xb.to(DEVICE), yb_d1.to(DEVICE), yb_raw.to(DEVICE), yb_jump.to(DEVICE)
            )
            opt.zero_grad()
            d_pred, r_pred, j_pred = model(xb)
            loss_d = ((d_pred - yb_d1) ** 2 * w).mean()
            loss_r = mae(r_pred, yb_raw)
            loss_j = bce(j_pred.view(-1), yb_jump)
            loss = 0.8 * loss_d + 0.1 * loss_r + 0.1 * loss_j
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} - loss: {total / len(ds):.6f}")

    # compute final bias
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb_d1, _, _ in dl:
            xb, yb_d1 = xb.to(DEVICE), yb_d1.to(DEVICE)
            d_pred, _, _ = model(xb)
            preds.append(d_pred.cpu().numpy())
            trues.append(yb_d1.cpu().numpy())
    bias = (np.concatenate(preds) - np.concatenate(trues)).mean(0)
    np.save(model.bias_path, bias.astype(np.float32))
    print("Saved bias to", model.bias_path)


# ============================================================
# MAIN — sweep PCA sizes and report R²
# ============================================================

if __name__ == "__main__":
    scorer = ScorerStepByStep(f"{CURRENT_DIR}/../../datasets/train.parquet")
    df = scorer.dataset
    cols = scorer.features

    # pca_k = 0 means your original model (no PCA features)
    pca_list = [0, 4, 8, 16, 32]
    summary = []

    for pca_k in pca_list:
        tag = f"pca{pca_k}" if pca_k > 0 else "base"

        print("\n" + "=" * 72)
        print(f"TRAINING / SCORING FOR PCA_K = {pca_k}")
        print("=" * 72)

        feat, d1, raw, jump = build_features_and_targets(df, cols, pca_k=pca_k, pca_tag=tag)
        input_dim = feat.shape[1]

        weights_path = os.path.join(CURRENT_DIR, f"lstm_h256_{tag}.pt")
        bias_path = os.path.join(CURRENT_DIR, f"delta_bias_h256_{tag}.npy")

        # Always start fresh per your request
        if os.path.exists(weights_path):
            os.remove(weights_path)
        if os.path.exists(bias_path):
            os.remove(bias_path)

        model = PredictionModel(
            hidden_size=256,
            input_dim=input_dim,
            pca_tag=None if pca_k == 0 else tag,
            weights_path=weights_path,
            bias_path=bias_path,
        )

        train_model(model, feat, d1, raw, jump, epochs=4)

        torch.save(model.state_dict(), weights_path)
        print("💾 Saved weights to", weights_path)

        # reload clean for scoring
        model = PredictionModel(
            hidden_size=256,
            input_dim=input_dim,
            pca_tag=None if pca_k == 0 else tag,
            weights_path=weights_path,
            bias_path=bias_path,
        )
        model.eval()
        results = scorer.score(model)

        mean_r2 = float(results["mean_r2"])
        summary.append((pca_k, mean_r2))

        print(f"\n📊 RESULTS for PCA_K = {pca_k}")
        print(f"Mean R²: {mean_r2:.6f}")
        print("First 5 features:")
        for f in cols[:5]:
            print(f"  {f}: {results[f]:.6f}")

    print("\n" + "=" * 60)
    print("PCA sweep summary (offline R²):")
    print("PCA_k\tMean_R2")
    for pca_k, r2 in summary:
        print(f"{pca_k}\t{r2:.6f}")
    print("=" * 60)
