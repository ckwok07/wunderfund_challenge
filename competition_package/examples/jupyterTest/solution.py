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
# 32-dim causal next-row predictor (no peeking)
# ============================================================

class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.dim = 32
        self.hidden_size = 64
        self.num_layers = 1

        # Keep state across timesteps during prediction
        self.current_seq_ix = None
        self.h = None
        self.c = None

        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.dim)

        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location=DEVICE))

        self.to(DEVICE)

    def forward(self, x, h=None, c=None):
        if h is None:
            out, (h, c) = self.lstm(x)
        else:
            out, (h, c) = self.lstm(x, (h, c))
        y = self.fc(out[:, -1, :])
        return y, h, c

    # ========================================================
    # ONLINE PREDICTION — identical logic to your working script
    # ========================================================
    @torch.no_grad()
    def predict(self, dp: DataPoint):
        x_t = torch.tensor(dp.state, dtype=torch.float32).view(1, 1, self.dim).to(DEVICE)

        # reset hidden state when new seq begins
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.h = None
            self.c = None

        # forward 1 timestep
        y, self.h, self.c = self.forward(x_t, self.h, self.c)

        # only output when scorer requests it
        if not dp.need_prediction:
            return None

        return y.squeeze(0).squeeze(0).detach().cpu().numpy()


# ============================================================
# TRAINING — EXACT same logic you used (x = v[:-1], y = v[1:])
# ============================================================

def train_model(model, df, num_epochs=4, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    features = df.columns[3:35]   # 32 raw inputs

    for epoch in range(num_epochs):
        total_loss = 0.0

        for seq_id, seq_df in df.groupby("seq_ix"):
            seq_df = seq_df.sort_values("step_in_seq")
            X_np = seq_df[features].to_numpy(np.float32)
            if len(X_np) < 2:
                continue

            x_np = X_np[:-1]
            y_np = X_np[1:]

            x = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE)  # (1, T-1, 32)
            y = torch.from_numpy(y_np).unsqueeze(0).to(DEVICE)  # (1, T-1, 32)

            optimizer.zero_grad()
            pred, _, _ = model.forward(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(CURRENT_DIR, "lstm_weights.pt"))
    print("Saved weights!")


# ============================================================
# MAIN ENTRY
# ============================================================

if __name__ == "__main__":
    scorer = ScorerStepByStep(f"{CURRENT_DIR}/../../datasets/train.parquet")
    df = scorer.dataset

    # Train
    model = PredictionModel()
    print("Training model...")
    train_model(model, df, num_epochs=4, lr=1e-3)

    # Evaluate
    print("Evaluating...")
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R²: {results['mean_r2']:.6f}")
    for f in scorer.features:
        print(f"{f}: {results[f]:.6f}")

    print("\nDone.")
