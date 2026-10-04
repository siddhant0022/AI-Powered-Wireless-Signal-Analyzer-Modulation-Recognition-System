
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
