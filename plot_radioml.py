import os

import matplotlib.pyplot as plt
import numpy as np

from dataset_loader import load_dataset

data = load_dataset("train", dataset="radioml")

modulation = "QPSK"  # Try BPSK, QAM16, 8PSK, etc.
snr_db = 18           # RadioML SNR levels range from -20 to +18 dB

matches = np.flatnonzero(
    (data["mod_name"] == modulation) & (data["snr"] == snr_db)
)
if len(matches) == 0:
    raise ValueError(f"No {modulation} frames found at {snr_db} dB")

frame = data["X"][matches[0]]
i_samples, q_samples = frame[0], frame[1]
sample_index = np.arange(len(i_samples))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(sample_index, i_samples, label="I")
axes[0].plot(sample_index, q_samples, label="Q")
axes[0].set(title=f"{modulation} waveform at {snr_db} dB", xlabel="Sample", ylabel="Amplitude")
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(i_samples, q_samples, s=12, alpha=0.7)
axes[1].set(title="Raw I/Q samples", xlabel="I", ylabel="Q")
axes[1].axis("equal")
axes[1].grid(True)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/radioml_{modulation.lower()}_{snr_db}db.png", dpi=150)
plt.show()