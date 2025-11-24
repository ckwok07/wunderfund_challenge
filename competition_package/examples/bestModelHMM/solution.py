import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

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
# Simple 4-state HMM on |delta1| magnitude (1D)
# ============================================================

def _gaussian_pdf_1d(x, mean, var):
    """x: (N,), mean: (K,), var: (K,) → returns (N, K) densities."""
    x = x[:, None]          # (N,1)
    mean = mean[None, :]    # (1,K)
    var = var[None, :]      # (1,K)
    var = np.maximum(var, 1e-6)
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    diff = x - mean
    expo = -0.5 * (diff * diff / var)
    return coeff * np.exp(expo)   # (N,K)


def fit_gmm_for_hmm(mag, K=4, n_iter=10):
    """
    Fit a 1D Gaussian mixture on magnitudes mag (N,) with K components.
    Returns mixture weights, means, variances.
    """
    mag = mag.astype(np.float64)
    N = mag.shape[0]

    # init means from quantiles
    qs = np.quantile(mag, np.linspace(0.0, 1.0, K + 1)[1:-1])
    means = np.concatenate([
        [mag.min()],
        qs,
        [mag.max()]
    ])
    if means.shape[0] > K:
        means = means[:K]
    means = means.astype(np.float64)

    vars_ = np.full(K, np.var(mag) + 1e-3, dtype=np.float64)
    weights = np.full(K, 1.0 / K, dtype=np.float64)

    for _ in range(n_iter):
        # E-step
        dens = _gaussian_pdf_1d(mag, means, vars_)    # (N,K)
        resp = dens * weights[None, :]
        resp_sum = resp.sum(axis=1, keepdims=True) + 1e-12
        resp /= resp_sum                               # normalize → responsibilities

        # M-step
        Nk = resp.sum(axis=0) + 1e-12                 # (K,)
        weights = Nk / N
        means = (resp * mag[:, None]).sum(axis=0) / Nk
        diff2 = (mag[:, None] - means[None, :]) ** 2
        vars_ = (resp * diff2).sum(axis=0) / Nk
        vars_ = np.maximum(vars_, 1e-6)

    return weights.astype(np.float32), means.astype(np.float32), vars_.astype(np.float32)


def estimate_hmm_transitions(seq_ix, mag, weights, means, vars_, K=4):
    """
    Take the fitted GMM (weights, means, vars) on |delta1| and:
      - hard-assign each t to a state z_t = argmax_k p(k|m_t)
      - estimate initial state distribution π
      - estimate transition matrix A
    """
    N = mag.shape[0]
    seq_ix = seq_ix.astype(np.int64)

    # responsibilities from GMM
    dens = _gaussian_pdf_1d(mag.astype(np.float64),
                            means.astype(np.float64),
                            vars_.astype(np.float64))   # (N,K)
    resp = dens * weights[None, :]
    resp_sum = resp.sum(axis=1, keepdims=True) + 1e-12
    resp /= resp_sum
    z = resp.argmax(axis=1)   # hard state indices (N,)

    # initial distribution: first state per sequence
    unique_seqs, first_indices = np.unique(seq_ix, return_index=True)
    first_states = z[first_indices]
    pi_counts = np.bincount(first_states, minlength=K).astype(np.float64)
    if pi_counts.sum() == 0:
        pi = np.full(K, 1.0 / K, dtype=np.float32)
    else:
        pi = (pi_counts / pi_counts.sum()).astype(np.float32)

    # transitions
    A_counts = np.ones((K, K), dtype=np.float64) * 1e-3  # smoothing
    for i in range(N - 1):
        if seq_ix[i + 1] != seq_ix[i]:
            continue
        si = z[i]
        sj = z[i + 1]
        A_counts[si, sj] += 1.0

    A = A_counts / A_counts.sum(axis=1, keepdims=True)
    return pi.astype(np.float32), A.astype(np.float32)


def compute_hmm_state_probs(seq_ix, mag, pi, A, means, vars_, K=4):
    """
    Run forward filtering p(z_t | m_1..t) for each sequence, using only past data.
    Returns state_probs_all: (N, K).
    """
    N = mag.shape[0]
    seq_ix = seq_ix.astype(np.int64)
    mag = mag.astype(np.float64)

    state_probs = np.zeros((N, K), dtype=np.float32)

    # precompute emissions per timestep per state
    dens = _gaussian_pdf_1d(mag, means.astype(np.float64), vars_.astype(np.float64))   # (N,K)

    # process each sequence separately
    unique_seqs = np.unique(seq_ix)
    for s in unique_seqs:
        idx = np.where(seq_ix == s)[0]
        if idx.size == 0:
            continue
        alpha = pi.astype(np.float64)  # initial belief

        for t in idx:
            # emission p(m_t | z_t=k)
            b = dens[t]  # (K,)
            # predict step: alpha_pred = alpha @ A
            alpha_pred = alpha @ A.astype(np.float64)
            # update step: multiply by emission
            alpha_new = alpha_pred * b
            ssum = alpha_new.sum()
            if ssum <= 0:
                alpha_new = np.full_like(alpha_new, 1.0 / K)
            else:
                alpha_new /= ssum
            alpha = alpha_new
            state_probs[t] = alpha.astype(np.float32)

    return state_probs  # (N,K)


# ============================================================
# Model definition (LSTM + HMM state features)
# ============================================================

class PredictionModel(nn.Module):

    def __init__(self,
                 hidden_size: int = 256,
                 input_dim: int = 164,   # 32+32+32+32+32+4
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

        # HMM setup
        self.hmm_K = 4
        self.hmm_alpha = None  # current belief over 4 hidden states

        # try to load HMM parameters (trained offline in main)
        try:
            self.hmm_pi = np.load(os.path.join(CURRENT_DIR, "hmm_pi_k4.npy")).astype(np.float32)
            self.hmm_A = np.load(os.path.join(CURRENT_DIR, "hmm_A_k4.npy")).astype(np.float32)
            self.hmm_means = np.load(os.path.join(CURRENT_DIR, "hmm_means_k4.npy")).astype(np.float32)
            self.hmm_vars = np.load(os.path.join(CURRENT_DIR, "hmm_vars_k4.npy")).astype(np.float32)
            print("Loaded HMM params (4-state) from disk.")
        except Exception:
            # fallback: simple defaults (still legal, but weaker if HMM files missing)
            print("HMM param files not found — using fallback uniform HMM.")
            self.hmm_pi = np.full(self.hmm_K, 1.0 / self.hmm_K, dtype=np.float32)
            self.hmm_A = np.full((self.hmm_K, self.hmm_K), 1.0 / self.hmm_K, dtype=np.float32)
            self.hmm_means = np.linspace(0.0, 1.0, self.hmm_K, dtype=np.float32)
            self.hmm_vars = np.ones(self.hmm_K, dtype=np.float32)

        # online buffers
        self.current_seq_ix = None
        self.sequence_history = []  # list of feature vectors (input_dim,)
        self.last_state = None
        self.delta_buffer = []
        self.raw_buffer = []

        # LSTM + heads
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.delta_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(self.output_dim)]
        )
        self.fc_raw = nn.Linear(self.hidden_size, self.raw_dim)
        self.head_jump = nn.Linear(self.hidden_size, 1)

        # weights / bias paths
        if weights_path is None:
            self.weights_path = os.path.join(CURRENT_DIR, "lstm_h256.pt")
        else:
            self.weights_path = weights_path

        if bias_path is None:
            self.bias_path = os.path.join(CURRENT_DIR, "delta_bias_h256.npy")
        else:
            self.bias_path = bias_path

        # load trained LSTM weights if present
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
        self.hmm_alpha = None

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

    def _update_hmm_alpha(self, mag_val: float):
        """
        One step of HMM forward filtering on |delta1| magnitude.
        mag_val: scalar float (current |delta1|).
        Uses only past state self.hmm_alpha and HMM params.
        """
        K = self.hmm_K
        pi = self.hmm_pi
        A = self.hmm_A
        means = self.hmm_means
        vars_ = np.maximum(self.hmm_vars, 1e-6)

        # previous belief
        if self.hmm_alpha is None:
            alpha_prev = pi.astype(np.float64)
        else:
            alpha_prev = self.hmm_alpha.astype(np.float64)

        # emission p(m | z=k)
        x = float(mag_val)
        diff = x - means.astype(np.float64)
        var = vars_.astype(np.float64)
        log_b = -0.5 * (diff * diff / var + np.log(2.0 * np.pi * var))
        m = log_b.max()
        b = np.exp(log_b - m)  # rescale for stability

        # predict
        alpha_pred = alpha_prev @ A.astype(np.float64)  # (K,)
        alpha = alpha_pred * b
        ssum = alpha.sum()
        if ssum <= 0:
            alpha = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            alpha /= ssum

        self.hmm_alpha = alpha.astype(np.float32)
        return self.hmm_alpha

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        if self.current_seq_ix != dp.seq_ix:
            self._reset(dp.seq_ix)

        raw = dp.state.astype(np.float32)
        if self.last_state is None:
            d1 = np.zeros_like(raw)
        else:
            d1 = raw - self.last_state
        self.last_state = raw

        rm_delta, vol_delta = self._update_rm_delta(d1)
        rm_raw = self._update_rm_raw(raw)
        norm_d1 = d1 / (vol_delta + 1e-6)
        norm_d1 = np.clip(norm_d1, -8, 8)

        # magnitude for HMM
        mag_val = float(np.linalg.norm(d1))
        hmm_state = self._update_hmm_alpha(mag_val)  # (4,)

        # feature vector: [raw, delta1, rm_delta, rm_raw, norm_d1, hmm_state]
        feat = np.concatenate([raw, d1, rm_delta, rm_raw, norm_d1, hmm_state], axis=0)
        self.sequence_history.append(feat.astype(np.float32))
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        if not dp.need_prediction:
            return None

        x = torch.tensor(
            np.stack(self.sequence_history, axis=0),
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        delta_pred, _, _ = self.forward(x)
        delta_pred = delta_pred.cpu().numpy().reshape(-1)

        if hasattr(self, "bias") and self.bias is not None:
            delta_pred = delta_pred - self.bias

        return raw + delta_pred


# ============================================================
# Training utilities (offline)
# ============================================================

def build_features_and_targets(df, cols, rm_window=5, hmm_K=4):
    """
    Build:
      - feats: (N, 164) = [raw, delta1, rm_delta1, rm_raw, norm_delta1, hmm_state_probs(4)]
      - d1:    (N, 32)  = delta1
      - raw:   (N, 32)  = raw level
      - jump:  (N,)     = binary jump label (top 20% |delta1|)
    Also:
      - fits a 4-state HMM on |delta1| magnitude
      - saves HMM params to disk for online use.
    """
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw = df[cols].to_numpy(np.float32)
    seq_ix = df["seq_ix"].to_numpy()

    N, dim = raw.shape
    d1 = np.zeros_like(raw, np.float32)
    rm_delta = np.zeros_like(raw, np.float32)
    rm_raw = np.zeros_like(raw, np.float32)
    vol = np.zeros_like(raw, np.float32)
    mag = np.zeros(N, dtype=np.float32)

    # per-sequence computation of deltas & rolling stats
    for seq_val, df_seq in df.groupby("seq_ix"):
        idx = df_seq.index.to_numpy()
        raw_seq = raw[idx]        # (T,32)
        T = raw_seq.shape[0]

        d_seq = np.zeros_like(raw_seq, np.float32)
        if T > 1:
            d_seq[1:] = raw_seq[1:] - raw_seq[:-1]

        rm_delta_seq = np.zeros_like(raw_seq, np.float32)
        rm_raw_seq = np.zeros_like(raw_seq, np.float32)
        vol_seq = np.zeros_like(raw_seq, np.float32)

        buf_d = []
        buf_r = []
        for t in range(T):
            buf_d.append(d_seq[t])
            if len(buf_d) > rm_window:
                buf_d = buf_d[-rm_window:]
            arr_d = np.stack(buf_d, axis=0)
            rm_delta_seq[t] = arr_d.mean(axis=0)
            vol_seq[t] = arr_d.std(axis=0)

            buf_r.append(raw_seq[t])
            if len(buf_r) > rm_window:
                buf_r = buf_r[-rm_window:]
            arr_r = np.stack(buf_r, axis=0)
            rm_raw_seq[t] = arr_r.mean(axis=0)

        d1[idx] = d_seq
        rm_delta[idx] = rm_delta_seq
        rm_raw[idx] = rm_raw_seq
        vol[idx] = vol_seq
        mag[idx] = np.linalg.norm(d_seq, axis=1).astype(np.float32)

    norm = d1 / (vol + 1e-6)
    norm = np.clip(norm, -8, 8)

    # jump labels = top 20% magnitude
    thresh = np.quantile(mag, 0.8)
    jump = (mag > thresh).astype(np.float32)

    # ----- Fit 4-state HMM on |delta1| magnitude -----
    gmm_weights, hmm_means, hmm_vars = fit_gmm_for_hmm(mag, K=hmm_K, n_iter=10)
    hmm_pi, hmm_A = estimate_hmm_transitions(seq_ix, mag, gmm_weights, hmm_means, hmm_vars, K=hmm_K)
    hmm_state_probs = compute_hmm_state_probs(seq_ix, mag, hmm_pi, hmm_A, hmm_means, hmm_vars, K=hmm_K)

    # save HMM params for online model usage
    np.save(os.path.join(CURRENT_DIR, "hmm_pi_k4.npy"), hmm_pi)
    np.save(os.path.join(CURRENT_DIR, "hmm_A_k4.npy"), hmm_A)
    np.save(os.path.join(CURRENT_DIR, "hmm_means_k4.npy"), hmm_means)
    np.save(os.path.join(CURRENT_DIR, "hmm_vars_k4.npy"), hmm_vars)
    print("Saved 4-state HMM params (pi, A, means, vars) to disk.")

    # feats = [raw, d1, rm_delta, rm_raw, norm, hmm_state_probs]
    feats = np.concatenate(
        [raw, d1, rm_delta, rm_raw, norm, hmm_state_probs],
        axis=1
    )  # (N, 32+32+32+32+32+4=164)

    return feats.astype(np.float32), d1.astype(np.float32), raw.astype(np.float32), jump.astype(np.float32)


def make_sequences(feat, d1, raw, jump, seq_len=32):
    X, yd, yr, yj = [], [], [], []
    N = feat.shape[0]
    for i in range(N - seq_len):
        X.append(feat[i:i + seq_len])
        yd.append(d1[i + seq_len])
        yr.append(raw[i + seq_len])
        yj.append(jump[i + seq_len])
    return (
        np.stack(X),
        np.stack(yd),
        np.stack(yr),
        np.array(yj, np.float32)
    )


def train_model(model, feat, d1, raw, jump, epochs=4, seq_len=32, lr=1e-3, batch=64):
    X, yd, yr, yj = make_sequences(feat, d1, raw, jump, seq_len)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.tensor(yd), torch.tensor(yr), torch.tensor(yj)
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

    # compute final bias in delta space
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
# MAIN — train once if needed, otherwise just score
# ============================================================

if __name__ == "__main__":
    scorer = ScorerStepByStep(f"{CURRENT_DIR}/../../datasets/train.parquet")
    df = scorer.dataset
    cols = scorer.features

    feat, d1, raw, jump = build_features_and_targets(df, cols)

    weights_path = os.path.join(CURRENT_DIR, "lstm_h256.pt")
    bias_path = os.path.join(CURRENT_DIR, "delta_bias_h256.npy")

    if os.path.exists(weights_path) and os.path.exists(bias_path):
        print("⚡ Found existing 256-weights — skipping training")
        model = PredictionModel(256, input_dim=feat.shape[1],
                                weights_path=weights_path,
                                bias_path=bias_path)
    else:
        print("🚀 No 256-weights found — training now...")
        model = PredictionModel(256, input_dim=feat.shape[1],
                                weights_path=weights_path,
                                bias_path=bias_path)
        train_model(model, feat, d1, raw, jump, epochs=4)
        torch.save(model.state_dict(), weights_path)
        print("💾 Saved weights to", weights_path)
        model = PredictionModel(256, input_dim=feat.shape[1],
                                weights_path=weights_path,
                                bias_path=bias_path)

    model.eval()
    results = scorer.score(model)
    print("\nMean R²:", results["mean_r2"])
    for f in cols:
        print(f, results[f])
