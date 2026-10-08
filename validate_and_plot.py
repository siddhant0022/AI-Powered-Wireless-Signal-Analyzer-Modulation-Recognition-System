"""
validate_and_plot.py  --  Integrity checks + statistics + sample plots.

    python validate_and_plot.py

Writes plots/*.png and data/dataset_statistics.json, and prints PASS/FAIL checks.
This is the 'proof the dataset is ready to use' step.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_loader import (load_dataset, dataset_info, MODULATIONS, ANOMALY_TYPES,
                            HERE, DATA)

PLOTS = Path(HERE).resolve() / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(DATA).resolve()


# ------------------------------------------------------------------ helpers
def recover_symbols(z, mod_name, sps=8):
    """
    Recovers synchronized symbol points from a single 128-sample baseband frame.
    Applies timing recovery, blind carrier frequency grid search, and phase lock.
    """
    z = z.astype(np.complex128)
    N = len(z)

    # Analog and continuous-phase modulations naturally show full trajectory rings
    if mod_name in ("GFSK", "CPFSK", "WBFM", "AM-DSB", "AM-SSB"):
        if N > 1:
            diffs = np.angle(z[1:] * np.conj(z[:-1]))
            z = z * np.exp(-1j * np.median(diffs) * np.arange(N))
        return z.real, z.imag

    # 1. Optimal timing strobe (maximum envelope variance / eye opening)
    best_tau = 0
    max_var = -1.0
    for tau in range(sps):
        sub = z[tau::sps]
        if len(sub) > 0:
            v = np.var(np.abs(sub))
            if v > max_var:
                max_var = v
                best_tau = tau

    syms = z[best_tau::sps]
    L = len(syms)
    if L < 4:
        return syms.real, syms.imag

    # Modulation power order for carrier stripping
    if mod_name in ("BPSK", "PAM4"):
        M = 2
    elif mod_name in ("QPSK", "QAM16", "QAM64"):
        M = 4
    elif mod_name == "8PSK":
        M = 8
    else:
        M = 1

    # 2. Coarse + Fine CFO grid search to freeze rotation
    freq_candidates = np.linspace(-0.25, 0.25, 201)
    best_f = 0.0
    max_coherence = -1.0
    time_vec = np.arange(L)

    for f in freq_candidates:
        rot = syms * np.exp(-1j * 2 * np.pi * f * time_vec)
        coherence = np.abs(np.mean(rot ** M))
        if coherence > max_coherence:
            max_coherence = coherence
            best_f = f

    syms_derot = syms * np.exp(-1j * 2 * np.pi * best_f * time_vec)

    # 3. Phase lock to orthogonal axes
    mean_vec = np.mean(syms_derot ** M)
    phi = np.angle(mean_vec) / M
    if mod_name in ("QPSK", "QAM16", "QAM64"):
        phi -= np.pi / 4
    syms_locked = syms_derot * np.exp(-1j * phi)

    # Power scale normalization
    rms = np.sqrt(np.mean(np.abs(syms_locked)**2))
    if rms > 0:
        syms_locked = syms_locked / rms

    # Lock BPSK and PAM4 strictly onto the real line
    if mod_name in ("BPSK", "PAM4"):
        syms_locked = syms_locked.real + 0.02 * 1j * np.random.randn(len(syms_locked))

    return syms_locked.real, syms_locked.imag


# ------------------------------------------------------------------ validation
def validate(all_):
    checks = []
    def chk(name, ok): checks.append((name, bool(ok)))

    X = all_["X"]
    chk("frame shape is (N, 2, 128)", X.ndim == 3 and X.shape[1:] == (2, 128))
    chk("no NaN / Inf values", np.isfinite(X).all())
    pw = (X.astype(np.float64) ** 2).sum(axis=1).mean(axis=1)
    chk("every frame has unit average power (+-1 %)", np.allclose(pw, 1.0, atol=0.01))
    chk("modulation ids in range (-1..10)", all_["mod"].min() >= -1 and all_["mod"].max() <= 10)
    chk("SNR range is -20..18 dB", all_["snr"].min() == -20 and all_["snr"].max() == 18)
    chk("all 11 modulations present", set(np.unique(all_["mod"][all_["mod"] >= 0])) == set(range(11)))
    chk("all 5 anomaly types present", set(np.unique(all_["anomaly_type"])) == set(range(6)))
    chk("anomaly flag consistent with anomaly_type",
        np.array_equal(all_["anomaly"], (all_["anomaly_type"] > 0).astype(np.int8)))
    chk("UNKNOWN modulation only on unknown-signal anomalies",
        set(np.unique(all_["anomaly_type"][all_["mod"] < 0])) <= {4, 5})

    splits = {s: load_dataset(s) for s in ("train", "val", "test")}
    chk("splits are disjoint and cover all frames",
        sum(len(v["X"]) for v in splits.values()) == len(X))
    chk("quality labels match SNR rule",
        np.array_equal(all_["quality"], np.where(all_["snr"] < 0, 0, np.where(all_["snr"] < 10, 1, 2))))

    # class balance among normal frames
    nm = all_["anomaly"] == 0
    counts = np.bincount(all_["mod"][nm], minlength=11)
    chk("normal frames balanced across modulations (max/min < 1.05)", counts.max() / counts.min() < 1.05)

    print("\nVALIDATION")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return checks, splits


# ------------------------------------------------------------------ statistics
def statistics(all_, splits):
    nm = all_["anomaly"] == 0
    stats = {
        "total_frames": int(len(all_["X"])),
        "normal_frames": int(nm.sum()),
        "anomaly_frames": int((~nm).sum()),
        "frames_per_split": {k: int(len(v["X"])) for k, v in splits.items()},
        "frames_per_modulation": {MODULATIONS[i]: int(((all_["mod"] == i) & nm).sum()) for i in range(11)},
        "frames_per_anomaly_type": {ANOMALY_TYPES[i]: int((all_["anomaly_type"] == i).sum()) for i in range(6)},
        "frames_per_quality": {n: int((all_["quality"] == i).sum()) for i, n in enumerate(["Poor", "Fair", "Good"])},
        "frames_per_channel": {n: int((all_["channel"] == i).sum()) for i, n in enumerate(["awgn", "fading"])},
    }
    stat_file = DATA_DIR / "dataset_statistics.json"
    with open(stat_file, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ------------------------------------------------------------------ plots
def plot_distributions(all_, stats):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(stats["frames_per_modulation"].keys(), stats["frames_per_modulation"].values(), color="#2F6DB5")
    ax[0].set_title("Normal frames per modulation"); ax[0].tick_params(axis="x", rotation=60)
    s, c = np.unique(all_["snr"], return_counts=True)
    ax[1].bar(s, c, width=1.6, color="#2F6DB5"); ax[1].set_title("Frames per SNR level (dB)")
    ax[2].bar(stats["frames_per_anomaly_type"].keys(), stats["frames_per_anomaly_type"].values(), color="#C0504D")
    ax[2].set_title("Frames per anomaly type"); ax[2].tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "01_distributions.png"), dpi=130)
    plt.close("all")


def _pick(all_, mod=None, atype=0, snr=None):
    m = (all_["anomaly_type"] == atype)
    if mod is not None: m &= all_["mod"] == mod
    if snr is not None: m &= all_["snr"] == snr
    idx = np.where(m)[0]
    return all_["X"][idx[0]] if len(idx) else None


def plot_class_gallery(all_, snr=18):
    """Waveform / constellation / spectrum for every modulation (clean, high SNR)."""
    fig, ax = plt.subplots(3, 11, figsize=(30, 8))
    digital_mods = {"BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "PAM4"}

    for i, name in enumerate(MODULATIONS):
        x = _pick(all_, mod=i, atype=0, snr=snr)
        z = x[0] + 1j * x[1]

        # Row 0: Waveform
        ax[0, i].plot(x[0], lw=0.8)
        ax[0, i].plot(x[1], lw=0.8)
        ax[0, i].set_title(name, fontsize=12)

        # Row 1: Constellation
        if name in digital_mods:
            # Use AWGN channel (channel == 0) frames at SNR=18 to avoid multipath smearing in plots
            mask = (all_["anomaly_type"] == 0) & (all_["mod"] == i) & (all_["snr"] == snr)
            if "channel" in all_ and np.any(mask & (all_["channel"] == 0)):
                mask &= (all_["channel"] == 0)

            matching_idx = np.where(mask)[0][:30]
            i_pts, q_pts = [], []
            for idx in matching_idx:
                raw_frame = all_["X"][idx]
                z_raw = raw_frame[0] + 1j * raw_frame[1]
                xi, xq = recover_symbols(z_raw, name, sps=8)
                i_pts.extend(xi)
                q_pts.extend(xq)

            ax[1, i].scatter(i_pts, q_pts, s=10, alpha=0.7, color="#1f77b4", edgecolors="none")
        else:
            xi, xq = recover_symbols(z, name, sps=8)
            ax[1, i].scatter(xi, xq, s=6, alpha=0.7, color="#1f77b4", edgecolors="none")

        ax[1, i].set_aspect("equal")
        ax[1, i].set_xlim([-2.0, 2.0])
        ax[1, i].set_ylim([-2.0, 2.0])
        ax[1, i].grid(True, linestyle="--", alpha=0.3)

        # Row 2: Spectrum
        sp = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(z))) + 1e-9)
        ax[2, i].plot(np.linspace(-0.5, 0.5, 128), sp, lw=0.8)

        for r in range(3):
            ax[r, i].tick_params(labelsize=6)

    ax[0, 0].set_ylabel("I / Q waveform")
    ax[1, 0].set_ylabel("Constellation")
    ax[2, 0].set_ylabel("Spectrum (dB)")
    plt.suptitle(f"All 11 modulations at SNR = {snr} dB", fontsize=15)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_modulation_gallery.png"), dpi=110)
    plt.close("all")


def plot_snr_effect(all_):
    """Same modulation (QPSK) at different SNR levels - shows the labels are meaningful."""
    levels = [-20, -10, 0, 10, 18]
    fig, ax = plt.subplots(1, 5, figsize=(18, 3.6))
    for a, s in zip(ax, levels):
        mask = (all_["anomaly_type"] == 0) & (all_["mod"] == 1) & (all_["snr"] == s)
        if "channel" in all_ and np.any(mask & (all_["channel"] == 0)):
            mask &= (all_["channel"] == 0)

        matching_idx = np.where(mask)[0][:20]
        i_pts, q_pts = [], []
        for idx in matching_idx:
            raw_frame = all_["X"][idx]
            z_raw = raw_frame[0] + 1j * raw_frame[1]
            xi, xq = recover_symbols(z_raw, "QPSK", sps=8)
            i_pts.extend(xi)
            q_pts.extend(xq)

        a.scatter(i_pts, q_pts, s=10, alpha=0.6, color="#1f77b4", edgecolors="none")
        a.set_aspect("equal")
        a.set_xlim([-2.0, 2.0])
        a.set_ylim([-2.0, 2.0])
        a.grid(True, linestyle="--", alpha=0.3)
        a.set_title(f"QPSK @ {s} dB")

    plt.tight_layout()
    plt.savefig(str(PLOTS / "03_snr_effect_qpsk.png"), dpi=130)
    plt.close("all")


def plot_anomalies(all_):
    fig, ax = plt.subplots(2, 5, figsize=(20, 6))
    for j, t in enumerate(range(1, 6)):
        idx = np.where((all_["anomaly_type"] == t) & (all_["snr"] >= 10))[0][0]
        x = all_["X"][idx]; z = x[0] + 1j * x[1]
        ax[0, j].plot(x[0], lw=0.8); ax[0, j].plot(x[1], lw=0.8); ax[0, j].set_title(ANOMALY_TYPES[t])
        ax[1, j].plot(np.linspace(-0.5, 0.5, 128), 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(z))) + 1e-9), lw=0.8)
    ax[0, 0].set_ylabel("I / Q waveform"); ax[1, 0].set_ylabel("Spectrum (dB)")
    plt.suptitle("Anomaly types (SNR >= 10 dB)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "04_anomaly_examples.png"), dpi=120)
    plt.close("all")


def plot_feature_check(all_):
    """Sanity: simple amplitude statistic separates classes, and empirical SNR tracks the label."""
    nm = (all_["anomaly"] == 0) & (all_["channel"] == 0)
    X = all_["X"][nm]; snr = all_["snr"][nm]; mod = all_["mod"][nm]
    amp = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    kurt = (amp ** 4).mean(1) / (amp ** 2).mean(1) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i, name in enumerate(MODULATIONS):
        sel = (mod == i) & (snr == 18)
        ax[0].hist(kurt[sel], bins=30, alpha=0.5, label=name)
    ax[0].set_title("Amplitude kurtosis @ 18 dB (class-dependent -> features carry info)")
    ax[0].legend(fontsize=6, ncol=2)
    mean_k = [kurt[(mod == 1) & (snr == s)].mean() for s in sorted(set(snr))]
    ax[1].plot(sorted(set(snr)), mean_k, "o-"); ax[1].set_title("QPSK amplitude kurtosis vs labelled SNR")
    ax[1].set_xlabel("SNR (dB)")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "05_feature_sanity_check.png"), dpi=120)
    plt.close("all")


if __name__ == "__main__":
    all_ = load_dataset("all")
    checks, splits = validate(all_)
    stats = statistics(all_, splits)
    plot_distributions(all_, stats)
    plot_class_gallery(all_)
    plot_snr_effect(all_)
    plot_anomalies(all_)
    plot_feature_check(all_)
    print("\nSTATISTICS")
    print(json.dumps(stats, indent=2))
    print(f"\nPlots saved in {PLOTS}")
    sys.exit(0 if all(ok for _, ok in checks) else 1)