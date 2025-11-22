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
            print(f"🔁 Loading saved LSTM weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=DEVICE)
            self.load_state_dict(state_dict)
        else:
            print("No saved LSTM weights found — using random initialization")


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
        # Reset state on new sequence
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.sequence_history = []
            self._fill_count = 0

            # Create LSTM hidden state (kept across timesteps)
            self._h = torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)
            self._c = torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)

        # Convert state to tensor ONCE
        x = torch.tensor(dp.state, dtype=torch.float32, device=DEVICE).view(1, 1, -1)  # (1,1,dim)

        # Run a single-timestep update instead of full sequence
        self.eval()
        out, (self._h, self._c) = self.lstm(x, (self._h, self._c))  # persistent state
        y = self.fc(out[:, 0, :])  # use hidden output

        self._fill_count += 1

        # Scorer calls predict() for many rows before requesting prediction
        if not dp.need_prediction or self._fill_count < 32:  # warm-up period
            return None

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
    data_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    scorer = ScorerStepByStep(data_file)
    model = PredictionModel()   # may load weights

    train_array = scorer.dataset[scorer.features].values.astype(np.float32)
    weights_path = os.path.join(CURRENT_DIR, "lstm_weights.pt")

    # Train only if no weights exist
    if not os.path.exists(weights_path):
        print("\n Training model — first run")
        train_model(
            model,
            train_array=train_array,
            num_epochs=3,
            lr=1e-3,
            batch_size=64,
        )
        print("\n💾 Saving trained weights...")
        torch.save(model.state_dict(), weights_path)
    else:
        print("\n⏭ Using pretrained weights — skipping training")

    # Evaluate
    print("\n📊 Scoring model...")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")