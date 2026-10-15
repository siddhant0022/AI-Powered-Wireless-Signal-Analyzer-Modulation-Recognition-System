"""Compact 1D CNN for modulation recognition and optional SNR estimation."""
import torch
from torch import nn


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
    def __init__(self, n_mod=11, use_snr=False, widths=(32, 64, 128), dropout=0.3):
        super().__init__()
        w1, w2, w3 = widths
        self.use_snr = use_snr
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

    def forward(self, x):
        features = self.features(x)
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)
        shared = self.shared(pooled)
        logits = self.modulation_head(shared)
        if not self.use_snr:
            return logits
        return logits, self.snr_head(shared)


if __name__ == "__main__":
    model = MultiTaskCNN()
    output = model(torch.zeros(4, 2, 128))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Modulation output: {tuple(output.shape)}")
    print(f"Parameters: {parameter_count:,}")
