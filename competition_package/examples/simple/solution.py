import os
import sys
import numpy as np

import pandas as pd

import torch
print("cuda ava", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
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

        self.input_dim = 64
        self.output_dim = 32

        self.hidden_size = 64
        self.num_layers = 1

        self.current_seq_ix = None
        self.sequence_history = []

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Fully-connected layer that maps hidden state → prediction
        self.heads = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(32)])
        self.target_index = 0

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
        device=x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,device=device)

        out, _ = self.lstm(x, (h0, c0))      # (batch_size, seq_len, hidden_size)


        last = out[:, -1, :]   # (batch, hidden_size)

        if self.training:
            # during training use SINGLE target → set at train time
            k = self.target_index
            return self.heads[k](last)    # (batch, 1)
        else:
            # during scoring → predict all 32 at once
            outputs = [head(last) for head in self.heads]   # list of (batch,1)
            return torch.cat(outputs, dim=1)                # (batch, 32)

    
    @torch.no_grad()
    def predict(self, dp: DataPoint):

        # Reset history when a new sequence starts
        if self.current_seq_ix != dp.seq_ix:
            self.current_seq_ix = dp.seq_ix
            self.sequence_history = []

        # Add the current timestep to history
        self.sequence_history.append(dp.state.astype(np.float32))

        # If scorer doesn't request a prediction yet, return None
        if not dp.need_prediction:
            return None

        seq = np.stack(self.sequence_history, axis=0)        # (seq_len, 32)

        rm5 = pd.DataFrame(seq).rolling(window=5, min_periods=1).mean().to_numpy().astype(np.float32)   # (seq_len, 32)

        inp = np.concatenate([seq, rm5], axis=1)             # (seq_len, 64)

        x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(DEVICE)


        # Run through the model
        self.eval()
        y = self.forward(x)  # (1, dim)
        self.train()

        return y.detach().cpu().numpy().reshape(-1)

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 32):

    """
    Turn a 2D array (N, dim) into many (input_seq, target) pairs:
    - input_seq: shape (seq_len, dim)
    - target:    shape (dim,) = next timestep after the window
    """
    X_seqs, y_seqs = [], []
    for i in range(len(X) - seq_len):
        X_seqs.append(X[i:i+seq_len])      # window
        y_seqs.append(y[i+seq_len])        # next step
    X_out = np.stack(X_seqs, axis=0)  # (num_samples, seq_len, dim)
    y_out = np.stack(y_seqs, axis=0)  # (num_samples, dim)
    return X_out, y_out


def train_model(
    model: PredictionModel,
    train_array: np.ndarray,
    num_epochs: int = 256,
    seq_len: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    model.train()
    model.to(DEVICE)
    X_all = train_array.astype(np.float32)  # (N, 64)
    y_raw = X_all[:, :32]
    loss_fn = nn.MSELoss()

    X, y = make_sequences(X_all, y_raw, seq_len=seq_len)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_index, (xb, yb) in enumerate(loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            k = batch_index % 32
            model.target_index = k

            optimizer.zero_grad()

            preds = model(xb)       # (batch, 1)
            preds = preds.view(-1)

            target = yb[:, k]

            loss = loss_fn(preds, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}")

def tune_hyperparameters(train_array):
    param_grid = {
        "num_epochs": [32, 64],
        "lr": [1e-3, 5e-4],
        "batch_size": [256, 384, 512],
        "seq_len": [32, 64],
    }

    best_score = -999
    best_params = None
    best_model = None


    for num_epochs in param_grid["num_epochs"]:
        for lr in param_grid["lr"]:
            for batch_size in param_grid["batch_size"]:
                for seq_len in param_grid["seq_len"]:

                    print("\n==========================================")
                    print(f"Testing: epochs={num_epochs}, lr={lr}, batch={batch_size}, seq={seq_len}")
                    print("==========================================")

                    # reinitialize model for each test run
                    model = PredictionModel()

                    train_model(
                        model,
                        train_array=train_array,
                        num_epochs=num_epochs,
                        lr=lr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                    )

                    results = scorer.score(model)
                    score = results["mean_r2"]
                    print(f"Mean R² = {score:.6f}")

                    if score > best_score:
                        best_score = score
                        best_params = (num_epochs, lr, batch_size, seq_len)
                        best_model = model
                        print("best hyperparameters found")

    print("\n============ TUNING COMPLETE ============")
    print(f"Best Mean R²: {best_score:.6f}")
    print(f"Best Params: epochs={best_params[0]}, lr={best_params[1]}, batch={best_params[2]}, seq={best_params[3]}")
    return best_model, best_params





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

    # first 3 are metadata / not predictive
    metadata_cols = df.columns[:3].tolist()

    # remaining columns are the true raw features (32 of them)
    raw_features = df.columns[3:].tolist()

    # add rolling mean only for raw columns
    for col in raw_features:
        df[f"{col}_rm5"] = df[col].rolling(window=5, min_periods=1).mean()

    # final input feature set (raw + rolling)
    rm5_features = [f"{col}_rm5" for col in raw_features]
    new_features = raw_features + rm5_features   # should be 32 + 32 = 64

    # build training array
    train_array = df[new_features].values.astype(np.float32)



    best_model, best_params = tune_hyperparameters(train_array)

    # Save trained weights to file in the same folder as solution.py
    torch.save(model.state_dict(), os.path.join(CURRENT_DIR, "lstm_weights.pt"))


    # Evaluate our solution
    results = scorer.score(best_model)   

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
