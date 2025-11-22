import os
import sys
import numpy as np

import pandas as pd

import torch
import torch.nn as nn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint, ScorerStepByStep


class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.dim = 32
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
        self.fc = nn.Linear(self.hidden_size, self.dim)

        self.current_seq_ix = None
        self.sequence_history = []

        weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))      # (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])         # (batch_size, dim)
        return out

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
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, dim)

        # Run through the model
        y = self.forward(x)  # (1, dim)

        return y.numpy().reshape(-1)

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


def train_model(
    model: PredictionModel,
    train_array: np.ndarray,
    num_epochs: int = 3,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    """
    Train the model offline on the full dataset *before* scoring.
    """
    model.train()

    X, y = make_sequences(train_array, seq_len=seq_len)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):      # <-- num_epochs is used here
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}")



if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    # Create and test our model
    model = PredictionModel()

    print("Testing simple model with moving average...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    train_array = scorer.dataset[scorer.features].values.astype(np.float32)

    train_model(
        model,
        train_array=train_array,
        num_epochs=3,
        lr=1e-3,
        batch_size=64,
    )

    # Save trained weights to file in the same folder as solution.py
    torch.save(model.state_dict(), os.path.join(CURRENT_DIR, "lstm_weights.pt"))


    # Evaluate our solution
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
