"""Stacked LSTM for multi-step blood glucose forecasting."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import rmse, mae


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class GlucoseLSTM(nn.Module):
    """
    Stacked LSTM that maps a (batch, seq_len, n_features) input to
    (batch, horizon) glucose predictions.

    Parameters
    ----------
    n_features  : number of input features per timestep
    hidden_size : LSTM hidden dimension (default 64)
    num_layers  : stacked LSTM depth (default 2)
    horizon     : number of future glucose steps to predict (default 6)
    dropout     : dropout probability between LSTM layers (default 0.2)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: GlucoseLSTM,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[GlucoseLSTM, dict]:
    """
    Train a GlucoseLSTM with Adam and early stopping.

    Parameters
    ----------
    X_train / X_val : (n_samples, seq_len, n_features)  float32-compatible
    y_train / y_val : (n_samples, horizon)               normalised glucose
    model           : uninitialised or pre-built GlucoseLSTM instance
    lr              : Adam learning rate (default 1e-3)
    max_epochs      : hard epoch cap (default 100)
    patience        : early-stopping patience in epochs (default 10)

    Returns
    -------
    best_model : GlucoseLSTM  — weights from the epoch with lowest val loss
    history    : dict with keys 'train_loss' and 'val_loss' (lists of floats)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = _make_loader(X_train, y_train)
    val_loader = _make_loader(X_val, y_val)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        train_loss = epoch_loss / len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss /= len(X_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    return model, history


def evaluate(
    model: GlucoseLSTM,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    y_mean: float,
    y_std: float,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate a trained GlucoseLSTM; inverse-transforms predictions to mg/dL.

    Parameters
    ----------
    X_test      : (n_samples, seq_len, n_features)  normalised
    y_test_raw  : (n_samples, horizon)               original scale [mg/dL]
    y_mean, y_std : normalisation stats for the glucose column (from scaler)

    Returns
    -------
    (rmse_val, mae_val, predictions_mg_dl)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()

    predictions = pred_norm * y_std + y_mean
    return rmse(y_test_raw, predictions), mae(y_test_raw, predictions), predictions
