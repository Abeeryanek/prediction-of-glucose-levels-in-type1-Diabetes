"""
Seq2Seq LSTM Autoencoder for blood glucose forecasting.

Architecture follows Srivastava et al. (2015) "Unsupervised Learning of Video
Representations using LSTMs": the encoder compresses the input sequence into a
fixed-size latent vector; the decoder is a full LSTM (not a linear layer)
that is initialised from the latent vector and predicts the target sequence
autoregressively, feeding each predicted step as the next decoder input.

A bottleneck linear layer (hidden_size → latent_size) provides explicit
dimensionality reduction before the decoder is initialised.
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


class GlucoseSeq2Seq(nn.Module):
    """
    Seq2Seq LSTM Autoencoder for multi-step glucose forecasting.

    Encoder
    -------
    LSTM reads the full input window (batch, seq_len, n_features) and
    produces a final hidden state h_n of shape (num_layers, batch, hidden_size).

    Bottleneck
    ----------
    The last-layer hidden state is compressed:
        z = bottleneck(h_n[-1])          # (batch, latent_size)

    Decoder initialisation
    ----------------------
    The latent vector is expanded back to initialise the decoder hidden state:
        dec_h0 = z_to_dec(z).view(num_layers, batch, hidden_size)   # h_0
        dec_c0 = zeros_like(dec_h0)                                  # c_0

    Decoding (autoregressive)
    -------------------------
    The decoder receives the last known glucose value as its first input
    (x[:, -1:, 0:1]) and produces `horizon` predictions one step at a time,
    feeding each predicted value back as the next input. A linear output
    layer maps each decoder hidden state → one glucose value.

    Parameters
    ----------
    n_features  : number of input features per timestep
    hidden_size : LSTM hidden dimension (default 64)
    latent_size : bottleneck dimension (default 32 — half of hidden_size)
    num_layers  : stacked LSTM depth for both encoder and decoder (default 2)
    horizon     : number of future glucose steps to predict (default 6)
    dropout     : dropout between LSTM layers (default 0.2)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 2,
        horizon: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 64 → 32  (information bottleneck)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        # 32 → 64 * num_layers  (expand to initialise decoder h_0)
        self.z_to_dec = nn.Linear(latent_size, hidden_size * num_layers)

        self.decoder = nn.LSTM(
            input_size=1,           # receives one glucose value per step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        (batch, horizon)
        """
        batch = x.size(0)

        # ── Encode ──────────────────────────────────────────────────────────
        _, (h_n, _) = self.encoder(x)
        # h_n: (num_layers, batch, hidden_size) — take the top layer
        h_last = h_n[-1]                             # (batch, hidden_size)

        # ── Bottleneck ───────────────────────────────────────────────────────
        z = self.bottleneck(h_last)                  # (batch, latent_size)

        # ── Decoder initialisation ───────────────────────────────────────────
        dec_init = self.z_to_dec(z)                  # (batch, hidden_size * num_layers)
        # Reshape to (num_layers, batch, hidden_size) required by nn.LSTM
        h_0 = (
            dec_init
            .view(batch, self.num_layers, self.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c_0 = torch.zeros_like(h_0)

        # ── Autoregressive decoding ──────────────────────────────────────────
        # Seed: last known glucose value in the input window
        dec_input = x[:, -1:, 0:1]                  # (batch, 1, 1)
        h, c = h_0, c_0
        predictions: list[torch.Tensor] = []

        for _ in range(self.horizon):
            out, (h, c) = self.decoder(dec_input, (h, c))
            # out: (batch, 1, hidden_size)
            pred = self.output_layer(out[:, 0, :])   # (batch, 1)
            predictions.append(pred)
            dec_input = pred.unsqueeze(1)            # feed prediction as next input

        return torch.cat(predictions, dim=1)         # (batch, horizon)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: GlucoseSeq2Seq,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[GlucoseSeq2Seq, dict]:
    """
    Train a GlucoseSeq2Seq model with Adam and early stopping.

    Identical calling convention to lstm.train_model so that both models are
    interchangeable in the experiment loop.

    Parameters
    ----------
    X_train / X_val : (n_samples, seq_len, n_features)
    y_train / y_val : (n_samples, horizon)  normalised glucose
    model           : uninitialised GlucoseSeq2Seq instance
    lr              : Adam learning rate
    max_epochs      : hard epoch cap
    patience        : early-stopping patience

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
    model: GlucoseSeq2Seq,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    y_mean: float,
    y_std: float,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate a trained GlucoseSeq2Seq; inverse-transforms predictions to mg/dL.

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
