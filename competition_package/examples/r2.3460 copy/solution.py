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


class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.dim = 64
        self.output_dim = 32
        self.hidden_size = 64
        self.num_layers = 1

        self.current_seq_ix = None
        self.sequence_history = []

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Fully-connected layer that maps hidden state → prediction
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

        self.current_seq_ix = None
        self.sequence_history = []

        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=DEVICE)
            self.load_state_dict(state_dict)

        self.to(DEVICE)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,device = device)

        out, _ = self.lstm(x, (h0, c0))      # (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])         # (batch_size, dim)
        return out
    
    @torch.no_grad()
    def predict(self, dp: DataPoint):
        if self.dim is None:
            self.dim = dp.state.shape[0]
            self.lstm = nn.LSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, self.dim)

        # Reset history when a new sequence starts
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.sequence_history = []

        # Add the current timestep to history
        self.sequence_history.append(dp.state.astype(np.float32))

        # If scorer doesn't request a prediction yet, return None
        if not dp.need_prediction:
            return None

        # Build input tensor of shape (1, seq_len, dim)
        seq = np.stack(self.sequence_history, axis=0)            # (seq_len, dim)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, seq_len, dim)

        # Run through the model
        y = self.forward(x)  # (1, dim)

        return y.detach().cpu().numpy().reshape(-1)

def make_sequences(data: np.ndarray, seq_len: int = 32):
    """
    Turn a 2D array (N, dim) into many (input_seq, target) pairs:
    - input_seq: shape (seq_len, dim)
    - target:    shape (dim,) = next timestep after the window
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])      # window
        y.append(data[i+seq_len])        # next step
    X = np.stack(X, axis=0)  # (num_samples, seq_len, dim)
    y = np.stack(y, axis=0)  # (num_samples, dim)
    return X, y

def custom_loss(pred, target, alpha=0.1):
    """
    Loss that preserves amplitude:
    MSE + variance alignment
    """
    mse = torch.mean((pred - target) ** 2)

    # Amplitude / variance penalty
    pred_std = pred.std(dim=0)
    targ_std = target.std(dim=0)
    var_pen = torch.mean((pred_std - targ_std) ** 2)

    return mse + alpha * var_pen


def train_model(model, train_array, num_epochs=4, seq_len=64, lr=1.5e-3, batch_size=64):
    model.train()
    X, y = make_sequences(train_array, seq_len=seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y),
                                         batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()

            preds = model(xb)
            loss = custom_loss(preds, yb, alpha=0.1)  # ⬅ changed line

            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} loss={total/len(loader.dataset):.6f}")




if __name__ == "__main__":
    # Path to weights file
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")

    # Check existence of train file
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    # If no weights yet → train and save them
    if not os.path.exists(weights_path):
        print("No lstm_weights.pt found training model to generate weights...")

        model = PredictionModel()  # will NOT load weights because file doesn't exist yet

        print(f"Feature dimensionality: {scorer.dim}")
        print(f"Number of rows in dataset: {len(scorer.dataset)}")

        df = scorer.dataset.copy()

        raw = df.iloc[:, 3:35]  # raw variables
        rm5 = raw.rolling(window=5, min_periods=1).mean()

        train_array = raw.values.astype(np.float32)  # 32-dim input for compatibility


        train_model(
            model,
            train_array=train_array,
            num_epochs=12,
            lr=1.5e-3,
            batch_size=64,
            seq_len=64
        )

        # Save trained weights
        torch.save(model.state_dict(), weights_path)
        print("Training complete. Saved weights to lstm_weights.pt")

    else:
        print("Found existing lstm_weights.pt – skipping training.")

    # Now load model with weights and score
    model = PredictionModel()  # this time it WILL load lstm_weights.pt
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(len(scorer.features)):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Done scoring with LSTM model.")
    print("=" * 60)

