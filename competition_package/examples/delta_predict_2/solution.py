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

        # We know there are 32 raw features
        self.dim = 32
        self.hidden_size = 64
        self.num_layers = 1
        self.max_seq_len = 32  # how many timesteps of deltas we feed in

        # Online state
        self.current_seq_ix = None
        self.sequence_history = []  # list of delta vectors
        self.last_state = None      # last raw state seen

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Fully-connected layer that maps hidden state → delta prediction
        self.fc = nn.Linear(self.hidden_size, self.dim)

        # If we have previously trained weights, load them
        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=DEVICE)
            self.load_state_dict(state_dict)

        self.to(DEVICE)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, dim) of delta values
        returns: (batch_size, dim) predicted delta at next step
        """
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))      # (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])         # (batch_size, dim)
        return out

    def _reset_sequence_state(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.sequence_history = []
        self.last_state = None

    @torch.no_grad()
    def predict(self, dp: DataPoint):
        """
        Online prediction using only changes in variables.

        - We observe raw dp.state (shape (32,))
        - Convert it to delta vs last_state
        - Feed recent history of deltas into LSTM
        - LSTM predicts next delta
        - We convert predicted delta back to a raw prediction for scorer
        """

        # New sequence → reset state
        if self.current_seq_ix != dp.seq_ix:
            self._reset_sequence_state(dp.seq_ix)

        raw_state = dp.state.astype(np.float32)

        # Compute delta from last_state
        if self.last_state is None:
            delta = np.zeros_like(raw_state, dtype=np.float32)
        else:
            delta = raw_state - self.last_state

        # Update last_state to current raw observation
        self.last_state = raw_state

        # Append delta to history
        self.sequence_history.append(delta)
        if len(self.sequence_history) > self.max_seq_len:
            self.sequence_history = self.sequence_history[-self.max_seq_len:]

        # If scorer doesn't request a prediction yet, return None
        if not dp.need_prediction:
            return None

        # Build input tensor of shape (1, seq_len, dim)
        seq = np.stack(self.sequence_history, axis=0)  # (seq_len, dim)
        x = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Model predicts the next delta
        delta_pred = self.forward(x)  # (1, dim)
        delta_pred_np = delta_pred.detach().cpu().numpy().reshape(-1)

        # Convert predicted delta back to raw prediction: x_{t+1} = x_t + Δ
        raw_pred = raw_state + delta_pred_np

        return raw_pred


def make_sequences(data: np.ndarray, seq_len: int = 32):
    """
    Turn a 2D array (N, dim) into many (input_seq, target) pairs:
    - input_seq: shape (seq_len, dim)
    - target:    shape (dim,) = next timestep delta after the window
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])      # window
        y.append(data[i+seq_len])        # next step
    X = np.stack(X, axis=0)  # (num_samples, seq_len, dim)
    y = np.stack(y, axis=0)  # (num_samples, dim)
    return X, y


def train_model(
    model: PredictionModel,
    train_array: np.ndarray,
    num_epochs: int = 3,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    """
    Train the model offline on the full dataset of DELTAS *before* scoring.
    train_array is an (N, 32) array of delta rows.
    """
    model.train()
    model.to(DEVICE)

    X, y = make_sequences(train_array, seq_len=seq_len)
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


if __name__ == "__main__":
    # Path to train parquet
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    # Create model
    model = PredictionModel()

    print("Training LSTM on DELTAS (changes in variables) ...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    df = scorer.dataset
    features = scorer.features  # should be the 32 raw variables

    # Build delta array per sequence, then stack
    all_deltas = []
    for seq_ix, df_seq in df.groupby("seq_ix"):
        seq_vals = df_seq[features].values.astype(np.float32)
        if len(seq_vals) < 2:
            continue
        delta = np.diff(seq_vals, axis=0)  # shape (T-1, 32)
        all_deltas.append(delta)

    train_delta = np.concatenate(all_deltas, axis=0)  # (N_total-#seq, 32)
    print(f"Total delta rows for training: {train_delta.shape[0]}")

    # Train on deltas
    train_model(
        model,
        train_array=train_delta,
        num_epochs=3,
        lr=1e-3,
        batch_size=64,
    )

    # Save trained weights to file in the same folder as solution.py
    torch.save(model.state_dict(), os.path.join(CURRENT_DIR, "lstm_weights.pt"))
    print("Saved weights to lstm_weights.pt")

    # Evaluate our solution using scorer's online interface
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
