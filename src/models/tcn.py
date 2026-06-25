"""
Temporal Convolutional Network for blood glucose forecasting.

Architecture: 4 dilated residual blocks with dilations [1, 2, 4, 8],
causal (left-only) padding so no future information leaks into the past,
BatchNorm after each convolution, and a skip connection (with a 1×1
projection if channel dimensions differ).
"""

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


class _CausalConv1d(nn.Module):
    """
    1-D convolution with left-only (causal) padding.

    Standard nn.Conv1d pads both sides symmetrically. This module adds
    (kernel_size − 1) × dilation steps of zero-padding on the left only,
    then trims the right side of the output to restore the original length.
    This guarantees that position t cannot attend to positions > t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self._pad,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self._pad > 0:
            return out[:, :, : -self._pad]   # remove right-side padding
        return out


class _TCNBlock(nn.Module):
    """
    One dilated residual TCN block.

    Structure (per block)
    ---------------------
      CausalConv1d → BatchNorm → ReLU → Dropout
      CausalConv1d → BatchNorm → ReLU → Dropout
      + residual skip (1×1 conv if in_channels ≠ out_channels)
      → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 1×1 projection for the residual branch if channel counts differ
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + self.skip(x))


class GlucoseTCN(nn.Module):
    """
    4-level dilated TCN for multi-step glucose forecasting.

    Dilations  : [1, 2, 4, 8]   (exponential growth covers ≥ 12 timesteps
                                  of receptive field with kernel_size=3)
    num_filters: 64  per block
    kernel_size: 3

    Input  : (batch, seq_len, n_features)  — transposed to channel-first internally
    Output : (batch, horizon)
    """

    def __init__(
        self,
        n_features: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        horizon: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        blocks: list[nn.Module] = []
        in_ch = n_features
        for d in dilations:
            blocks.append(_TCNBlock(in_ch, num_filters, kernel_size, d, dropout))
            in_ch = num_filters
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(num_filters, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)          # (batch, num_filters, seq_len)
        return self.head(out[:, :, -1])  # use last time-step representation


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: GlucoseTCN,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[GlucoseTCN, dict]:
    """
    Train a GlucoseTCN with Adam and early stopping.

    Identical calling convention to lstm.train_model and autoencoder.train_model
    so all three deep learning models are interchangeable in the experiment loop.

    Parameters
    ----------
    X_train / X_val : (n_samples, seq_len, n_features)
    y_train / y_val : (n_samples, horizon)  normalised glucose
    model           : uninitialised GlucoseTCN instance
    lr              : Adam learning rate
    max_epochs      : hard epoch cap
    patience        : early-stopping patience in epochs

    Returns
    -------
    best_model, history   where history = {'train_loss': [...], 'val_loss': [...]}
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

    for _ in range(max_epochs):
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
    model: GlucoseTCN,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    y_mean: float,
    y_std: float,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate a trained GlucoseTCN; inverse-transforms predictions to mg/dL.

    Parameters
    ----------
    X_test      : (n_samples, seq_len, n_features)  normalised
    y_test_raw  : (n_samples, horizon)               original scale [mg/dL]
    y_mean, y_std : glucose normalisation stats from the pipeline scaler

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
