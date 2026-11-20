"""Compact 1D CNN for modulation recognition and optional SNR estimation.

With ``use_snr=False``, ``forward`` returns modulation logits of shape
``(B, n_mod)``. With ``use_snr=True``, it returns ``(mod_logits, snr_pred)``,
where ``snr_pred`` has shape ``(B, 1)`` and is scaled by 20.
"""
import os

import numpy as np
import torch
from torch import nn


def scale_snr_target(snr_db):
    """Scale SNR dB labels into the model's training range."""
    return snr_db / 20


def quality_from_snr(snr_db):
    """Map SNR in dB to 0=Poor, 1=Fair, or 2=Good; preserve scalar inputs."""
    if isinstance(snr_db, torch.Tensor):
        return torch.where(snr_db < 0, 0, torch.where(snr_db < 10, 1, 2))
    snr_db = np.asarray(snr_db)
    quality = np.where(snr_db < 0, 0, np.where(snr_db < 10, 1, 2))
    return int(quality) if quality.ndim == 0 else quality


class ResBlock(nn.Module):
    """Two same-width convolutions with a residual connection."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class MultiTaskCNN(nn.Module):
    def __init__(self, n_mod=11, use_snr=False, widths=(32, 64, 128), dropout=0.3,
                 snr_loss_beta=1.0, use_anomaly=False, use_anomaly_type=False):
        super().__init__()
        if snr_loss_beta <= 0:
            raise ValueError("snr_loss_beta must be positive")
        if use_anomaly_type and not use_anomaly:
            raise ValueError("use_anomaly_type requires use_anomaly=True")
        w1, w2, w3 = widths
        self.n_mod = n_mod
        self.use_snr = use_snr
        self.use_anomaly = use_anomaly
        self.use_anomaly_type = use_anomaly_type
        self.widths = tuple(widths)
        self.dropout = dropout
        self.snr_loss_beta = snr_loss_beta
        self.features = nn.Sequential(
            nn.Conv1d(2, w1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(w1),
            nn.ReLU(inplace=True),
            ResBlock(w1, kernel_size=5),
            nn.MaxPool1d(2),
            nn.Conv1d(w1, w2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(w2),
            nn.ReLU(inplace=True),
            ResBlock(w2, kernel_size=5),
            nn.MaxPool1d(2),
            nn.Conv1d(w2, w3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(w3),
            nn.ReLU(inplace=True),
            ResBlock(w3, kernel_size=3),
            nn.MaxPool1d(2),
        )
        self.shared = nn.Sequential(
            nn.Linear(2 * w3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.modulation_head = nn.Linear(128, n_mod)
        self.snr_head = (
            nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
            if use_snr else None
        )
        self.anomaly_head = nn.Linear(128, 1) if use_anomaly else None
        self.anomaly_type_head = (
            nn.Linear(128, 5) if use_anomaly_type else None
        )

    def forward(self, x):
        features = self.features(x)
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)
        shared = self.shared(pooled)
        logits = self.modulation_head(shared)
        if self.use_anomaly:
            outputs = {
                "mod_logits": logits,
                "anomaly_logit": self.anomaly_head(shared),
            }
            if self.use_snr:
                outputs["snr_pred"] = self.snr_head(shared)
            if self.use_anomaly_type:
                outputs["type_logits"] = self.anomaly_type_head(shared)
            return outputs
        if not self.use_snr:
            return logits
        return logits, self.snr_head(shared)

    def predict_snr_db(self, x):
        """Return SNR predictions in dB for an SNR-enabled model."""
        if not self.use_snr:
            raise RuntimeError("predict_snr_db requires a model constructed with use_snr=True")
        if self.use_anomaly:
            snr_pred = self(x)["snr_pred"]
        else:
            _, snr_pred = self(x)
        return snr_pred * 20

    def get_config(self):
        """Return the constructor options needed to recreate this model."""
        return {
            "n_mod": self.n_mod,
            "use_snr": self.use_snr,
            "widths": list(self.widths),
            "dropout": self.dropout,
            "snr_loss_beta": self.snr_loss_beta,
            "use_anomaly": self.use_anomaly,
            "use_anomaly_type": self.use_anomaly_type,
        }


def checkpoint_payload(model, **metadata):
    """Build a new checkpoint including model configuration and optional metadata."""
    reserved = {"config", "model_state_dict"}.intersection(metadata)
    if reserved:
        raise ValueError(f"metadata cannot replace reserved checkpoint fields: {sorted(reserved)}")
    return {
        **metadata,
        "config": model.get_config(),
        "model_state_dict": model.state_dict(),
    }


def load_model_checkpoint(path, map_location="cpu"):
    """Load a configured checkpoint or a legacy Stage A checkpoint without config."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{os.fspath(path)} is missing model_state_dict")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"{os.fspath(path)} has an invalid model config")
    model = MultiTaskCNN(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


if __name__ == "__main__":
    model = MultiTaskCNN()
    output = model(torch.zeros(4, 2, 128))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Modulation output: {tuple(output.shape)}")
    print(f"Parameters: {parameter_count:,}")
