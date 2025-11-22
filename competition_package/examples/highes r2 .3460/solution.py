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

        self.rm_buffer = []
        self.rm_sum = None


        self.dim = 64
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
        if not dp.need_prediction:
            return None

        # New sequence → reset recurrent state and rolling buffer
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.h = torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)
            self.c = torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)
            self.rm_buffer = []
            self.rm_sum = None

        # raw 32-dim state
        s = dp.state.astype(np.float32)

        # rolling mean over last 5 timesteps
        if self.rm_sum is None:
            self.rm_sum = s.copy()
            self.rm_buffer = [s]
            rm = s
        else:
            self.rm_sum += s
            self.rm_buffer.append(s)
            if len(self.rm_buffer) > 5:
                self.rm_sum -= self.rm_buffer.pop(0)
            rm = self.rm_sum / len(self.rm_buffer)

        # input is [raw, rm] = 64 dims
        inp = np.concatenate([s, rm], axis=0).astype(np.float32)
        x = torch.from_numpy(inp).view(1, 1, self.dim).to(DEVICE)

        out, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        y = self.fc(out[:, -1, :])
        return y.cpu().numpy().reshape(-1)



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
    model.to(DEVICE)

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
    # Path to weights file
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")

    # Check existence of train file
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    # If no weights yet → train and save them
    if not os.path.exists(weights_path):
        print("No lstm_weights.pt found – training model to generate weights...")

        model = PredictionModel()  # will NOT load weights because file doesn't exist yet

        print(f"Feature dimensionality: {scorer.dim}")
        print(f"Number of rows in dataset: {len(scorer.dataset)}")

        df = scorer.dataset.copy()

        raw = df.iloc[:, 3:35]  # raw variables
        rm5 = raw.rolling(window=5, min_periods=1).mean()

        train_array = np.concatenate([raw.values, rm5.values], axis=1).astype(np.float32)

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

