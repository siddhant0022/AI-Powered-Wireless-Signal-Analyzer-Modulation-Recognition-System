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
    channel       int8    (N,)         0 = awgn, 1 = fading, -1 = unavailable
    anomaly       int8    (N,)         0 = normal, 1 = anomaly
    anomaly_type  int8    (N,)         index into ANOMALY_TYPES (0 = none)
Options:
    normal_only=True   keep only normal frames (e.g. to train an anomaly detector)
    anomalies_only=True
    dataset="radioml"   load the converted RadioML 2016.10A dataset
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DATASET_FILES = {
    "simulated": "signal_dataset.npz",
    "radioml": "radioml_2016a.npz",
}

MODULATIONS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "PAM4",
               "GFSK", "CPFSK", "WBFM", "AM-DSB", "AM-SSB"]
ANOMALY_TYPES = ["none", "jamming_tone", "clipping", "impulsive_noise",
                 "unknown_ofdm", "unknown_chirp"]
QUALITY_NAMES = ["Poor", "Fair", "Good"]
CHANNEL_NAMES = ["awgn", "fading"]


def load_dataset(split="train", normal_only=False, anomalies_only=False, path=None,
                 dataset="simulated"):
    if split not in ("train", "val", "test", "all"):
        raise ValueError("split must be 'train', 'val', 'test', or 'all'")
    if dataset not in DATASET_FILES:
        raise ValueError(f"dataset must be one of: {', '.join(DATASET_FILES)}")
    path = path or os.path.join(DATA, DATASET_FILES[dataset])
    if not os.path.exists(path):
        if dataset == "radioml":
            action = "run: python import_radioml.py"
        else:
            action = "run: python build_dataset.py"
        raise FileNotFoundError(f"{path} not found - {action}")

    with np.load(path) as d:
        mask = np.ones(len(d["X"]), dtype=bool)
        if split != "all":
            mask &= d["split"] == {"train": 0, "val": 1, "test": 2}[split]

        if dataset == "radioml":
            mod, snr = d["mod"], d["snr"]
            channel = np.full(len(mod), -1, dtype=np.int8)
            anomaly = np.zeros(len(mod), dtype=np.int8)
            anomaly_type = np.zeros(len(mod), dtype=np.int8)
            quality = np.where(snr < 0, 0, np.where(snr < 10, 1, 2)).astype(np.int8)
        else:
            channel, anomaly = d["channel"], d["anomaly"]
            anomaly_type, quality = d["anomaly_type"], d["quality"]
            mod, snr = d["mod"], d["snr"]

        if normal_only:
            mask &= anomaly == 0
        if anomalies_only:
            mask &= anomaly == 1
        out = {
            "X": d["X"][mask],
            "mod": mod[mask],
            "snr": snr[mask],
            "channel": channel[mask],
            "anomaly": anomaly[mask],
            "anomaly_type": anomaly_type[mask],
            "quality": quality[mask],
        }
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
