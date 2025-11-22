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

        self.dim = None
        self.input_dim = 64
        self.output_dim = 32
        self.hidden_size = 256
        self.num_layers = 2
        self.seq_len = 96

        self.current_seq_ix = None
        self.sequence_history = []

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=.20,
            batch_first=True
        )

        # Fully-connected layer that maps hidden state → prediction
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

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
        rm5 = pd.DataFrame(seq).rolling(window=5, min_periods=1).mean().to_numpy()
        inp = np.concatenate([seq, rm5], axis=1)         # (seq_len, 64)
        x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        


        # Run through the model
        y = self.forward(x)  # (1, dim)

        return y.detach().cpu().numpy().reshape(-1)

def make_sequences(input64: np.ndarray, seq_len: int = 32):

    raw32= input64[:, :32]

    X, y = [], []
    for i in range(len(input64) - seq_len):
        X.append(input64[i:i+seq_len])      # window
        y.append(raw32[i+seq_len])        # next step
    X = np.stack(X, axis=0)  # (num_samples, seq_len, dim)
    y = np.stack(y, axis=0)  # (num_samples, dim)
    return X, y


@torch.no_grad()
def predict(self, dp: DataPoint):
    if self.dim is None:
        self.dim = dp.state.shape[0]

    # Reset internal buffers for a new sequence
    if self.current_seq_ix != dp.seq_ix:
        self.current_seq_ix = dp.seq_ix
        self.sequence_history = []
        self.rm_buffer = []         # store last 5 rows
        self.rm_sum = None          # running sum for fast RM5

    # Convert to float32
    s = dp.state.astype(np.float32)
    self.sequence_history.append(s)

    # ----- fast rolling mean of last 5 (O(1) runtime) -----
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

    # Build input window (last seq_len timesteps)
    seq_np = np.array(self.sequence_history[-self.seq_len:], dtype=np.float32)

    # Expand rm to match window shape
    rm_stack = np.tile(rm, (seq_np.shape[0], 1)).astype(np.float32)

    inp = np.concatenate([seq_np, rm_stack], axis=1)   # (T, 64)
    x = torch.from_numpy(inp).unsqueeze(0).to(DEVICE)  # (1, T, 64)

    y = self.forward(x)
    return y.detach().cpu().numpy().reshape(-1)



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

    df = scorer.dataset.copy()

    raw = df.iloc[:, 3:35] #raw variables
    rm5 = raw.rolling(window=5, min_periods=1).mean()

    train_array = np.concatenate([raw.values, rm5.values], axis=1).astype(np.float32)

    train_model(
        model,
        train_array=train_array,
        num_epochs=18,
        lr=1.5e-3,
        batch_size=64,
        seq_len=96
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


