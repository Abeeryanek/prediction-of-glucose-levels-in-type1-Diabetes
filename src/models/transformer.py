"""Transformer encoder for multi-step blood glucose forecasting.

Architecture (Vaswani et al. 2017):
  1. Input projection   nn.Linear(n_features, d_model)
  2. Positional encoding (sinusoidal)
  3. nn.TransformerEncoder  (batch_first=True, num_layers stacked layers)
  4. Last-timestep read  out[:, -1, :]
  5. Prediction head     nn.Linear(d_model, horizon)

References
----------
Vaswani et al. (2017) "Attention is All You Need"
Xiong et al. (2025) — Transformer on OhioT1DM, RMSE 19.33 mg/dL
Kalita & Mirza (2025) — multi-head attention on OhioT1DM, RMSE 16.57 mg/dL
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import rmse, mae


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GlucoseTransformer(nn.Module):
    """
    Transformer encoder that maps (batch, seq_len, n_features) → (batch, horizon).

    Every input timestep attends to every other timestep simultaneously via
    multi-head self-attention, giving the model a global view of the window
    without the sequential bottleneck of recurrent models.

    Parameters
    ----------
    n_features      : number of input features per timestep
    d_model         : embedding dimension (default 64; must be divisible by nhead)
    nhead           : number of attention heads (default 4)
    num_layers      : number of stacked TransformerEncoderLayer blocks (default 2)
    dim_feedforward : inner dimension of each feedforward sublayer (default 128)
    horizon         : number of future glucose steps to predict (default 6)
    dropout         : dropout rate applied inside attention and feedforward (default 0.1)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        horizon: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Step 1 — project variable-width feature space into fixed d_model space
        self.input_proj = nn.Linear(n_features, d_model)

        # Step 2 — inject sequence position information (no built-in ordering in attention)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        # Step 3 — stack of multi-head self-attention + feedforward blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Step 5 — linear readout from the last timestep's representation
        self.head = nn.Linear(d_model, horizon)

        # Xavier uniform for all weight matrices (dim > 1 skips bias vectors)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        (batch, horizon)
        """
        x = self.input_proj(x)       # (batch, seq_len, d_model)
        x = self.pos_enc(x)          # inject position info
        x = self.encoder(x)          # multi-head self-attention × num_layers
        return self.head(x[:, -1])   # last timestep → (batch, horizon)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: GlucoseTransformer,
    lr: float = 1e-3,
    max_epochs: int = 150,
    patience: int = 15,
    loss_fn=None,
) -> tuple[GlucoseTransformer, dict, int]:
    """
    Train a GlucoseTransformer with Adam and early stopping.

    Identical calling convention to lstm.train_model.

    Parameters
    ----------
    X_train / X_val : (n_samples, seq_len, n_features)
    y_train / y_val : (n_samples, horizon)  normalised glucose
    model           : GlucoseTransformer instance
    lr              : Adam learning rate (default 1e-3)
    max_epochs      : hard epoch cap (default 150)
    patience        : early-stopping patience (default 15)
    loss_fn         : optional callable(preds, target) -> scalar loss tensor.
                       Defaults to None, which uses plain nn.MSELoss() exactly
                       as before (non-breaking). Pass e.g.
                       src.training.losses.clinically_weighted_mse_scaled to
                       train with clinical sample weighting instead (un-scales
                       z-scored glucose to mg/dL internally so the clinical
                       thresholds apply to real glucose values).

    Returns
    -------
    best_model, history, actual_epochs
        history = {'train_loss': [...], 'val_loss': [...]}
        actual_epochs = number of epochs actually run before early stopping
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = _make_loader(X_train, y_train)
    val_loader   = _make_loader(X_val,   y_val)

    criterion = loss_fn if loss_fn is not None else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0

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
            best_weights  = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    actual_epochs = len(history["train_loss"])
    return model, history, actual_epochs


def evaluate(
    model: GlucoseTransformer,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    y_mean: float,
    y_std: float,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate a trained GlucoseTransformer; inverse-transforms predictions to mg/dL.

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
