"""
dataset_loader.py  --  One-line access to the dataset.

    from dataset_loader import load_dataset
    train = load_dataset("train")           # also "val", "test", or "all"
    X, y_mod, y_snr = train["X"], train["mod"], train["snr"]

Returned dict (all NumPy arrays, N = number of frames):
    X             float32 (N, 2, 128)  I/Q frame, unit average power
    mod           int16   (N,)         modulation id   (-1 = UNKNOWN signal)
    mod_name      str     (N,)         modulation name ("UNKNOWN" for unknown signals)
    snr           int16   (N,)         true SNR in dB  (used as the regression target)
    quality       int8    (N,)         0 = Poor, 1 = Fair, 2 = Good
    channel       int8    (N,)         0 = awgn, 1 = fading
    anomaly       int8    (N,)         0 = normal, 1 = anomaly
    anomaly_type  int8    (N,)         index into ANOMALY_TYPES (0 = none)
Options:
    normal_only=True   keep only normal frames (e.g. to train an anomaly detector)
    anomalies_only=True
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MODULATIONS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "PAM4",
               "GFSK", "CPFSK", "WBFM", "AM-DSB", "AM-SSB"]
ANOMALY_TYPES = ["none", "jamming_tone", "clipping", "impulsive_noise",
                 "unknown_ofdm", "unknown_chirp"]
QUALITY_NAMES = ["Poor", "Fair", "Good"]
CHANNEL_NAMES = ["awgn", "fading"]


def load_dataset(split="train", normal_only=False, anomalies_only=False, path=None):
    path = path or os.path.join(DATA, "signal_dataset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run:  python build_dataset.py")
    d = np.load(path)
    mask = np.ones(len(d["X"]), dtype=bool)
    if split != "all":
        mask &= d["split"] == {"train": 0, "val": 1, "test": 2}[split]
    if normal_only:
        mask &= d["anomaly"] == 0
    if anomalies_only:
        mask &= d["anomaly"] == 1
    out = {k: d[k][mask] for k in ("X", "mod", "snr", "channel", "anomaly",
                                   "anomaly_type", "quality")}
    out["mod_name"] = np.array([MODULATIONS[m] if m >= 0 else "UNKNOWN" for m in out["mod"]])
    return out


def dataset_info(path=None):
    path = path or os.path.join(DATA, "dataset_info.json")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        d = load_dataset(s)
        print(f"{s:5s}: X{d['X'].shape}  normal={int((d['anomaly']==0).sum())}  "
              f"anomalies={int(d['anomaly'].sum())}")
