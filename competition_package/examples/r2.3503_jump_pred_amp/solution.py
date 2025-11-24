# ============================================================
# LSTM WITH MODERATE AMPLITUDE LEARNING (D2) + MULTI-RUN TUNING
# ============================================================

print(">>> RUNNING THIS FILE <<<")

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


# ----------------------------- R² WEIGHTS -----------------------------
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
FEATURE_WEIGHTS = FEATURE_WEIGHTS / FEATURE_WEIGHTS.mean()


# ----------------------------- MODEL -----------------------------
class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.raw_dim = 32
        self.input_dim = 96          # [raw, delta1, rm(delta1)]
        self.output_dim = 32         # next delta1
        self.hidden_size = 96
        self.num_layers = 1
        self.max_seq_len = 32
        self.rm_window = 5

        # online state
        self.current_seq_ix = None
        self.sequence_history = []
        self.last_state = None
        self.delta_buffer = []

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # 32 independent delta heads
        self.delta_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(self.output_dim)]
        )

        # aux raw head
        self.fc_raw = nn.Linear(self.hidden_size, self.raw_dim)

        # jump timing head
        self.fc_jump = nn.Linear(self.hidden_size, 1)

        # amplitude head
        self.fc_amp = nn.Linear(self.hidden_size, 1)

        # bias vector (set after training)
        self.bias = None

        self.to(DEVICE)

    def forward(self, x):
        """
        x: (B, L, 96)
        returns:
            delta_pred: (B, 32)
            raw_pred:   (B, 32)
            jump_logit: (B, 1)
            amp_pred:   (B, 1)
        """
        out, _ = self.lstm(x)              # (B, L, H)
        last = out[:, -1, :]               # (B, H)

        delta_pred = torch.cat(
            [h(last) for h in self.delta_heads], dim=1
        )                                  # (B, 32)
        raw_pred = self.fc_raw(last)       # (B, 32)
        jump_logit = self.fc_jump(last)    # (B, 1)
        amp_pred = self.fc_amp(last)       # (B, 1)

        return delta_pred, raw_pred, jump_logit, amp_pred

    # ----------------------------- ONLINE PREDICTION -----------------------------
    @torch.no_grad()
    def predict(self, dp: DataPoint):
        # reset if new sequence
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.sequence_history = []
            self.last_state = None
            self.delta_buffer = []

        raw = dp.state.astype(np.float32)  # (32,)
        if self.last_state is None:
            delta = np.zeros_like(raw, dtype=np.float32)
        else:
            delta = raw - self.last_state
        self.last_state = raw

        # rolling mean over delta
        self.delta_buffer.append(delta)
        if len(self.delta_buffer) > self.rm_window:
            self.delta_buffer = self.delta_buffer[-self.rm_window:]
        rm = np.stack(self.delta_buffer, axis=0).mean(axis=0)  # (32,)

        feat = np.concatenate([raw, delta, rm], axis=0)        # (96,)
        self.sequence_history.append(feat)
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        if not dp.need_prediction:
            return None

        seq = np.stack(self.sequence_history, axis=0)          # (L, 96)
        x = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, L, 96)

        delta_pred, _, _, _ = self.forward(x)
        delta_np = delta_pred.cpu().numpy().reshape(-1)        # (32,)

        if self.bias is not None:
            delta_np = delta_np - self.bias

        return raw + delta_np


# ----------------------------- DATA PIPELINE -----------------------------
def build_features_and_targets(df: pd.DataFrame, feature_cols, rm_window: int = 5):
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    raw = df[feature_cols].to_numpy(np.float32)        # (N, 32)
    N = len(df)

    delta = np.zeros_like(raw, dtype=np.float32)
    rm = np.zeros_like(raw, dtype=np.float32)

    # per-sequence delta + rolling mean
    for _, seq in df.groupby("seq_ix"):
        idx = seq.index.to_numpy()
        r = raw[idx]          # (T, 32)
        T = r.shape[0]

        d = np.zeros_like(r, dtype=np.float32)
        if T > 1:
            d[1:] = r[1:] - r[:-1]

        buf = []
        rm_seq = np.zeros_like(r, dtype=np.float32)
        for t in range(T):
            buf.append(d[t])
            if len(buf) > rm_window:
                buf = buf[-rm_window:]
            rm_seq[t] = np.stack(buf, axis=0).mean(axis=0)

        delta[idx] = d
        rm[idx] = rm_seq

    # jump label from delta magnitude
    mag = np.linalg.norm(delta, axis=1)       # (N,)
    thresh = np.quantile(mag, 0.8)
    jumps = (mag > thresh).astype(np.float32)  # (N,)

    # amplitude target: magnitude on jumps, 0 otherwise
    amp = mag.copy().astype(np.float32)
    amp[jumps == 0] = 0.0

    feats = np.concatenate([raw, delta, rm], axis=1)   # (N, 96)
    return feats, delta, raw, jumps, amp


def make_sequences(feats, delta, raw, jumps, amp, L=32):
    X, Yd, Yr, Yj, Ya = [], [], [], [], []
    N = len(feats)
    for i in range(N - L):
        X.append(feats[i:i+L])
        Yd.append(delta[i+L])
        Yr.append(raw[i+L])
        Yj.append(jumps[i+L])
        Ya.append(amp[i+L])
    X = np.stack(X)
    Yd = np.stack(Yd)
    Yr = np.stack(Yr)
    Yj = np.array(Yj, dtype=np.float32)
    Ya = np.array(Ya, dtype=np.float32)
    return X, Yd, Yr, Yj, Ya


# ----------------------------- TRAINING -----------------------------
def train_model(
    model: PredictionModel,
    feats,
    delta,
    raw,
    jumps,
    amp,
    *,
    epochs=4,
    lr=1e-3,
    batch_size=64,
    seq_len=32,
    w_delta=0.75,
    w_raw=0.10,
    w_jump=0.13,
    w_amp=0.25,
):
    """
    Train with given loss weights:
      loss = w_delta * L_delta + w_raw * L_raw + w_jump * L_jump + w_amp * L_amp
    """
    print(f"\n🔧 Training with weights: "
          f"delta={w_delta}, raw={w_raw}, jump={w_jump}, amp={w_amp}")

    X, Yd, Yr, Yj, Ya = make_sequences(feats, delta, raw, jumps, amp, L=seq_len)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Yd, dtype=torch.float32),
        torch.tensor(Yr, dtype=torch.float32),
        torch.tensor(Yj, dtype=torch.float32),
        torch.tensor(Ya, dtype=torch.float32),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mae = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    w_feat = torch.tensor(FEATURE_WEIGHTS, dtype=torch.float32, device=DEVICE)

    for ep in range(epochs):
        total = 0.0
        for xb, y_delta, y_raw, y_jump, y_amp in dl:
            xb = xb.to(DEVICE)
            y_delta = y_delta.to(DEVICE)
            y_raw = y_raw.to(DEVICE)
            y_jump = y_jump.to(DEVICE)
            y_amp = y_amp.to(DEVICE)

            opt.zero_grad()
            d_pred, r_pred, j_logit, a_pred = model(xb)

            # delta loss (per-feature weighted MSE)
            loss_delta = ((d_pred - y_delta) ** 2 * w_feat).mean()
            # raw loss
            loss_raw = mae(r_pred, y_raw)
            # jump timing loss
            loss_jump = bce(j_logit.view(-1), y_jump)
            # amplitude loss only on jumps
            if (y_jump == 1).any():
                loss_amp = mae(a_pred.view(-1)[y_jump == 1], y_amp[y_jump == 1])
            else:
                loss_amp = 0.0

            loss = (
                w_delta * loss_delta +
                w_raw * loss_raw +
                w_jump * loss_jump +
                w_amp * loss_amp
            )

            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        print(f"Epoch {ep+1}/{epochs}  loss={total / len(ds):.6f}")

    # ---------------- amplitude diagnostics ----------------
    print("\n🔍 Amplitude diagnostics…")
    with torch.no_grad():
        amp_preds, jump_flags = [], []
        for xb, _, _, y_jump, _ in dl:
            xb = xb.to(DEVICE)
            _, _, _, a_pred = model(xb)
            amp_preds.append(a_pred.cpu().numpy().reshape(-1))
            jump_flags.append(y_jump.numpy())
    amp_preds = np.concatenate(amp_preds)
    jf = np.concatenate(jump_flags)

    if (jf == 1).any():
        avg_jump = amp_preds[jf == 1].mean()
    else:
        avg_jump = 0.0
    if (jf == 0).any():
        avg_non = amp_preds[jf == 0].mean()
    else:
        avg_non = 0.0

    ratio = avg_jump / max(abs(avg_non), 1e-8)
    print(f"avg_amp_pred_jump     = {avg_jump:.4f}")
    print(f"avg_amp_pred_nonjump  = {avg_non:.4f}")
    print(f"ratio (jump/nonjump)  = {ratio:.2f}x\n")

    # ---------------- bias correction in delta space ----------------
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, y_delta, _, _, _ in dl:
            xb = xb.to(DEVICE)
            y_delta = y_delta.to(DEVICE)
            d_pred, _, _, _ = model(xb)
            preds_all.append(d_pred.cpu().numpy())
            trues_all.append(y_delta.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    bias = (preds_all - trues_all).mean(axis=0)  # (32,)

    model.bias = bias  # set in-memory for scoring
    print("Bias vector set on model (no file save).")

    # return diagnostics in case you want to log later
    return {
        "avg_amp_pred_jump": float(avg_jump),
        "avg_amp_pred_non": float(avg_non),
        "ratio": float(ratio),
    }


# ----------------------------- MAIN: MULTI-CONFIG TUNING -----------------------------
if __name__ == "__main__":
    scorer = ScorerStepByStep(f"{CURRENT_DIR}/../../datasets/train.parquet")
    df = scorer.dataset
    feature_cols = scorer.features

    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(df)}")

    # Build features ONCE
    feats, delta, raw, jumps, amp = build_features_and_targets(df, feature_cols)
    print("Train feature shape:", feats.shape)
    print("Train delta shape:", delta.shape)
    print("Train raw shape:", raw.shape)
    print("Train jumps shape:", jumps.shape)
    print("Train amp shape:", amp.shape)

    # List of configs to try (all start from scratch)
    configs = [
        # name,   w_delta, w_raw, w_jump, w_amp
        ("A_base",      0.80,   0.10,   0.10,   0.02),
        ("B_jump_heavy",0.70,   0.10,   0.18,   0.02),
        ("C_amp_heavy", 0.70,   0.10,   0.10,   0.25),
        ("D_jump+amp",  0.65,   0.10,   0.18,   0.25),
    ]

    for name, w_delta, w_raw, w_jump, w_amp in configs:
        print("\n" + "=" * 70)
        print(f"🏁 RUN {name} — "
              f"delta={w_delta}, raw={w_raw}, jump={w_jump}, amp={w_amp}")
        print("=" * 70)

        # fresh model each time (no weight loading)
        model = PredictionModel()
        diag = train_model(
            model,
            feats,
            delta,
            raw,
            jumps,
            amp,
            epochs=4,
            lr=1e-3,
            batch_size=64,
            seq_len=32,
            w_delta=w_delta,
            w_raw=w_raw,
            w_jump=w_jump,
            w_amp=w_amp,
        )

        # score this model
        model.eval()
        results = scorer.score(model)

        print(f"\n📊 RESULTS for {name}")
        print(f"Mean R² across all features: {results['mean_r2']:.6f}")
        for i, feat_name in enumerate(scorer.features[:5]):
            print(f"  {i}: {results[feat_name]:.6f}")

        print("\nAmplitude diagnostics recap:")
        print(f"  avg_amp_pred_jump    = {diag['avg_amp_pred_jump']:.4f}")
        print(f"  avg_amp_pred_nonjump = {diag['avg_amp_pred_non']:.4f}")
        print(f"  ratio (jump/nonjump) = {diag['ratio']:.2f}x")

    print("\n" + "=" * 70)
    print("All configs finished. Pick the best R² + amplitude separation.")
    print("=" * 70)
