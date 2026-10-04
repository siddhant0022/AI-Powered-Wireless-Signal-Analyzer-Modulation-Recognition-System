"""
import_radioml.py  --  OPTIONAL: convert the public RadioML 2016.10A dataset into
the SAME format as our simulated dataset (so both can be used with load_dataset()).

1. Download RADIOML 2016.10A (RML2016.10a_dict.pkl) from https://www.deepsig.ai/datasets
2. python import_radioml.py path/to/RML2016.10a_dict.pkl

Output: data/radioml_2016a.npz   (fields: X, mod, snr, split  ; 220,000 frames)
Its 11 classes are: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
and are re-mapped to our class order (simulator.MODULATIONS).
"""
import os, pickle, sys
import numpy as np
from build_dataset import OUT_DIR, stratified_split
import simulator as sim

if len(sys.argv) != 2:
    sys.exit(__doc__)

with open(sys.argv[1], "rb") as f:
    d = pickle.load(f, encoding="latin1")                 # dict {(mod, snr): array(1000, 2, 128)}

X, mod, snr = [], [], []
for (m, s), arr in d.items():
    X.append(arr.astype(np.float32)); mod += [sim.MOD_ID[m]] * len(arr); snr += [s] * len(arr)
X = np.concatenate(X); mod = np.array(mod, np.int16); snr = np.array(snr, np.int16)

# cleaning: drop bad frames, normalise to unit average power
ok = np.isfinite(X).all(axis=(1, 2))
X, mod, snr = X[ok], mod[ok], snr[ok]
X /= np.sqrt((X.astype(np.float64) ** 2).sum(axis=1).mean(axis=1))[:, None, None]

rng = np.random.default_rng(2026)
split = stratified_split((mod.astype(int) + 1) * 1000 + (snr.astype(int) + 20), rng)
os.makedirs(OUT_DIR, exist_ok=True)
np.savez_compressed(os.path.join(OUT_DIR, "radioml_2016a.npz"), X=X, mod=mod, snr=snr, split=split)
print(f"Saved {len(X)} RadioML frames -> {OUT_DIR}/radioml_2016a.npz  (dropped {int((~ok).sum())} bad frames)")
