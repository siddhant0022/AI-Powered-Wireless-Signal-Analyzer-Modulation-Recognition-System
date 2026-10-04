"""
build_dataset.py  --  Generate, clean, label and split the dataset.

    python build_dataset.py                       # default: ~49k frames, ~50 MB
    python build_dataset.py --per-cell 100        # smaller / faster
    python build_dataset.py --per-cell 500 --seed 7

Pipeline (matches the October plan: collect/simulate -> clean -> label):
    1. SIMULATE : 11 modulations x 20 SNR levels x 2 channels (+ 5 anomaly types)
    2. CLEAN    : drop NaN/Inf and dead frames, unit-energy normalisation
    3. LABEL    : modulation, SNR, channel, anomaly flag + anomaly type
    4. SPLIT    : stratified 70 / 15 / 15  (train / val / test)
    5. SAVE     : data/signal_dataset.npz  +  data/labels.csv  +  data/dataset_info.json
"""
import argparse, csv, json, os, time
import numpy as np
import simulator as sim

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def make_normal_frame(mod, snr, channel, rng):
    x = sim.GENERATORS[mod](rng)
    x = sim.apply_channel(x, channel, rng)
    return sim.add_awgn(x, snr, rng)


def make_anomaly_frame(atype, rng):
    """Return (complex frame, base_modulation or 'UNKNOWN', snr_db, channel)."""
    snr = int(rng.choice(sim.SNR_LIST))
    channel = str(rng.choice(sim.CHANNELS))
    if atype in ("unknown_ofdm", "unknown_chirp"):
        x = sim.gen_ofdm(rng) if atype == "unknown_ofdm" else sim.gen_chirp(rng)
        x = sim.apply_channel(x, channel, rng)
        return sim.add_awgn(x, snr, rng), "UNKNOWN", snr, channel
    mod = str(rng.choice(sim.MODULATIONS))
    x = sim.apply_channel(sim.GENERATORS[mod](rng), channel, rng)
    x = sim.add_awgn(x, snr, rng)
    inject = {"jamming_tone": sim.inject_jamming_tone,
              "clipping": sim.inject_clipping,
              "impulsive_noise": sim.inject_impulses}[atype]
    return inject(x, rng), mod, snr, channel


def to_iq(x):
    """complex (128,) -> real (2, 128): row 0 = I, row 1 = Q."""
    return np.stack([x.real, x.imag]).astype(np.float32)


def clean(X, log):
    """Return boolean keep-mask and normalised X (unit RMS per frame)."""
    finite = np.isfinite(X).all(axis=(1, 2))
    energy = np.sqrt((X.astype(np.float64) ** 2).sum(axis=1).mean(axis=1))
    alive = energy > 1e-6
    keep = finite & alive
    log["removed_non_finite"] = int((~finite).sum())
    log["removed_zero_energy"] = int((finite & ~alive).sum())
    Xn = X.copy()
    Xn[keep] = X[keep] / energy[keep][:, None, None]       # unit average power per frame
    return keep, Xn


def stratified_split(strata, rng, fr=(0.70, 0.15, 0.15)):
    split = np.zeros(len(strata), dtype=np.int8)           # 0 train, 1 val, 2 test
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n_tr, n_va = int(round(fr[0] * len(idx))), int(round(fr[1] * len(idx)))
        split[idx[n_tr:n_tr + n_va]] = 1
        split[idx[n_tr + n_va:]] = 2
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=100,
                    help="frames per (modulation, SNR, channel) cell  [default 100]")
    ap.add_argument("--anomalies-per-type", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    X, mod_id, snr, chan, anom = [], [], [], [], []

    # 1. SIMULATE normal signals ------------------------------------------------
    for mod in sim.MODULATIONS:
        for ch in sim.CHANNELS:
            for s in sim.SNR_LIST:
                for _ in range(args.per_cell):
                    X.append(to_iq(make_normal_frame(mod, s, ch, rng)))
                    mod_id.append(sim.MOD_ID[mod]); snr.append(s)
                    chan.append(sim.CHAN_ID[ch]); anom.append(0)
        print(f"  simulated {mod:7s}  ({len(X):6d} frames so far)")

    # ... and anomalies ---------------------------------------------------------
    for atype in sim.ANOMALY_TYPES[1:]:
        for _ in range(args.anomalies_per_type):
            x, base, s, ch = make_anomaly_frame(atype, rng)
            X.append(to_iq(x))
            mod_id.append(-1 if base == "UNKNOWN" else sim.MOD_ID[base])
            snr.append(s); chan.append(sim.CHAN_ID[ch]); anom.append(sim.ANOM_ID[atype])
        print(f"  simulated anomaly: {atype}")

    X = np.stack(X)
    mod_id, snr = np.array(mod_id, np.int16), np.array(snr, np.int16)
    chan, anom = np.array(chan, np.int8), np.array(anom, np.int8)

    # 2. CLEAN -------------------------------------------------------------------
    log = {"frames_generated": int(len(X))}
    keep, X = clean(X, log)
    X, mod_id, snr, chan, anom = X[keep], mod_id[keep], snr[keep], chan[keep], anom[keep]
    log["frames_kept"] = int(len(X))
    print(f"  cleaning: {log}")

    # 3. LABEL (derived fields) ----------------------------------------------------
    # Signal-quality class derived from the true SNR  (0 = Poor, 1 = Fair, 2 = Good)
    quality = np.where(snr < 0, 0, np.where(snr < 10, 1, 2)).astype(np.int8)
    is_anom = (anom > 0).astype(np.int8)

    # 4. SPLIT (stratified by modulation x SNR x anomaly-type) --------------------
    strata = (mod_id.astype(int) + 1) * 10000 + (snr.astype(int) + 20) * 10 + anom
    split = stratified_split(strata, rng)

    # shuffle once so files are not ordered by class
    perm = rng.permutation(len(X))
    X, mod_id, snr, chan, anom, is_anom, quality, split = [
        a[perm] for a in (X, mod_id, snr, chan, anom, is_anom, quality, split)]

    # 5. SAVE ---------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "signal_dataset.npz"),
                        X=X, mod=mod_id, snr=snr, channel=chan, anomaly=is_anom,
                        anomaly_type=anom, quality=quality, split=split)

    with open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "split", "modulation", "snr_db", "channel", "quality",
                    "is_anomaly", "anomaly_type"])
        names_split = ["train", "val", "test"]
        names_q = ["Poor", "Fair", "Good"]
        for i in range(len(X)):
            w.writerow([i, names_split[split[i]],
                        "UNKNOWN" if mod_id[i] < 0 else sim.MODULATIONS[mod_id[i]],
                        int(snr[i]), sim.CHANNELS[chan[i]], names_q[quality[i]],
                        int(is_anom[i]), sim.ANOMALY_TYPES[anom[i]]])

    info = {
        "name": "AI Wireless Signal Analyzer - simulated I/Q dataset",
        "seed": args.seed, "frame_shape": [2, sim.FRAME_LEN],
        "samples_per_symbol": sim.SPS, "rrc_rolloff": sim.ROLLOFF,
        "modulations": sim.MODULATIONS, "snr_levels_db": sim.SNR_LIST,
        "channels": sim.CHANNELS, "anomaly_types": sim.ANOMALY_TYPES,
        "quality_rule": "Poor: SNR < 0 dB, Fair: 0-9 dB, Good: >= 10 dB",
        "split_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "n_total": int(len(X)),
        "n_per_split": {n: int((split == i).sum()) for i, n in enumerate(names_split)},
        "n_normal": int((is_anom == 0).sum()), "n_anomaly": int(is_anom.sum()),
        "cleaning_log": log, "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDONE  {len(X)} frames in {info['build_seconds']} s  ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
