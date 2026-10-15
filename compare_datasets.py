"""Compare RadioML 2016.10A against the simulated dataset.

Usage:
    python compare_datasets.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset_loader import DATA, MODULATIONS


ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
COMPARISON_FILE = Path(DATA) / "comparison.json"
KURTOSIS_MODULATIONS = ("BPSK", "QPSK", "QAM16", "WBFM")


def _load(path, simulated):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with np.load(path) as archive:
        required = {"X", "mod", "snr", "split"}
        if simulated:
            required |= {"channel", "anomaly"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        data = {key: archive[key] for key in required}

    if simulated:
        keep = (data["anomaly"] == 0) & (data["channel"] == 0) & (data["mod"] >= 0)
    else:
        keep = data["mod"] >= 0
    return {key: value[keep] for key, value in data.items()}


def _frames_at(data, modulation_id, snr):
    mask = (data["mod"] == modulation_id) & (data["snr"] == snr)
    return data["X"][mask]


def _mean_power_spectrum(frames):
    if not len(frames):
        raise ValueError("No frames available for the requested modulation and SNR")
    complex_frames = frames[:, 0] + 1j * frames[:, 1]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(complex_frames, axis=-1), axes=-1)) ** 2
    spectrum /= np.maximum(spectrum.sum(axis=-1, keepdims=True), 1e-20)
    mean_spectrum = spectrum.mean(axis=0)
    return 10 * np.log10(np.maximum(mean_spectrum, 1e-12))


def _mean_amplitude_kurtosis(frames):
    complex_frames = frames[:, 0] + 1j * frames[:, 1]
    power = np.mean(np.abs(complex_frames) ** 2, axis=1)
    fourth_moment = np.mean(np.abs(complex_frames) ** 4, axis=1)
    return float(np.mean(fourth_moment / np.maximum(power ** 2, 1e-20)))


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    radio = _load(os.path.join(DATA, "radioml_2016a.npz"), simulated=False)
    ours = _load(os.path.join(DATA, "signal_dataset.npz"), simulated=True)
    with open(os.path.join(DATA, "radioml_info.json"), encoding="utf-8") as f:
        radio_info = json.load(f)

    ours_power = np.mean(
        np.sum(ours["X"].astype(np.float64) ** 2, axis=1), axis=1
    )
    comparison = {
        "power": {
            "radioml_raw_before_normalization": radio_info["raw_average_power_per_frame"],
            "simulated_normalized_awgn_normal_frames": {
                "mean": float(ours_power.mean()),
                "min": float(ours_power.min()),
                "max": float(ours_power.max()),
            },
        },
        "spectra_snr_db": 18,
        "spectra": {},
        "amplitude_kurtosis_by_snr": {},
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    frequencies = np.fft.fftshift(np.fft.fftfreq(128))
    for class_id, modulation in enumerate(MODULATIONS):
        ax = axes.flat[class_id]
        r_frames = _frames_at(radio, class_id, 18)
        s_frames = _frames_at(ours, class_id, 18)
        r_spectrum = _mean_power_spectrum(r_frames)
        s_spectrum = _mean_power_spectrum(s_frames)
        comparison["spectra"][modulation] = {
            "radioml_mean_normalized_power_db": r_spectrum.tolist(),
            "simulated_mean_normalized_power_db": s_spectrum.tolist(),
            "radioml_frame_count": int(len(r_frames)),
            "simulated_frame_count": int(len(s_frames)),
        }
        ax.plot(frequencies, r_spectrum, label="RadioML", linewidth=1.2)
        ax.plot(frequencies, s_spectrum, label="Simulated", linewidth=1.2)
        ax.set_title(modulation)
        ax.grid(True, alpha=0.3)
    axes.flat[11].axis("off")
    axes[0, 0].legend()
    fig.supxlabel("Normalized frequency")
    fig.supylabel("Mean normalized power (dB)")
    fig.suptitle("RadioML vs simulated spectra at 18 dB")
    fig.tight_layout()
    fig.savefig(PLOTS / "06_spectra_radioml_vs_ours.png", dpi=140)
    plt.close(fig)

    snr_levels = sorted(int(s) for s in np.unique(radio["snr"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for modulation in KURTOSIS_MODULATIONS:
        class_id = MODULATIONS.index(modulation)
        radio_values, simulated_values = [], []
        for snr in snr_levels:
            r_frames = _frames_at(radio, class_id, snr)
            s_frames = _frames_at(ours, class_id, snr)
            radio_values.append(_mean_amplitude_kurtosis(r_frames) if len(r_frames) else None)
            simulated_values.append(
                _mean_amplitude_kurtosis(s_frames) if len(s_frames) else None
            )
        comparison["amplitude_kurtosis_by_snr"][modulation] = {
            "snr_db": snr_levels,
            "radioml": radio_values,
            "simulated": simulated_values,
        }
        ax.plot(snr_levels, radio_values, marker="o", label=f"{modulation} RadioML")
        ax.plot(snr_levels, simulated_values, marker=".", linestyle="--",
                label=f"{modulation} Simulated")
    ax.set(
        title="Mean amplitude kurtosis vs SNR",
        xlabel="Labelled SNR (dB)",
        ylabel="E|x|⁴ / (E|x|²)²",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "07_kurtosis_vs_snr.png", dpi=140)
    plt.close(fig)

    with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print("Power comparison (mean/min/max average power per frame):")
    print(f"  RadioML raw: {comparison['power']['radioml_raw_before_normalization']}")
    print(
        "  Simulated normal AWGN (unit normalized): "
        f"{comparison['power']['simulated_normalized_awgn_normal_frames']}"
    )
    print(f"Saved comparison plots to {PLOTS}")
    print(f"Saved comparison numbers to {COMPARISON_FILE}")


if __name__ == "__main__":
    main()
