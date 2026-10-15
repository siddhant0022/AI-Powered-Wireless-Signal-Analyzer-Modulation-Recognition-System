"""Convert RadioML 2016.10A into the project's normalised NPZ format.

Usage:
    python import_radioml.py path/to/RML2016.10a_dict.pkl
"""
import argparse
import json
import os
import pickle

import numpy as np

from build_dataset import OUT_DIR, stratified_split
import simulator as sim


def convert(input_path):
    """Validate and convert the trusted RadioML pickle at input_path."""
    with open(input_path, "rb") as source:
        data = pickle.load(source, encoding="latin1")
    if not isinstance(data, dict):
        raise ValueError("RadioML input must be a dict keyed by (modulation, SNR)")

    frames, mod_ids, snrs, frame_powers = [], [], [], []
    raw_powers = []
    seen_modulations, seen_snrs = set(), set()
    for key, values in data.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Invalid RadioML key: {key!r}")
        modulation, snr = key
        if isinstance(modulation, bytes):
            modulation = modulation.decode("latin1")
        if modulation not in sim.MOD_ID:
            raise ValueError(f"Unsupported RadioML modulation: {modulation!r}")

        values = np.asarray(values)
        if values.ndim != 3 or values.shape[1:] != (2, sim.FRAME_LEN):
            raise ValueError(
                f"RadioML entry {key!r} must have shape (N, 2, {sim.FRAME_LEN}); "
                f"got {values.shape}"
            )
        power = np.mean(np.sum(values.astype(np.float64) ** 2, axis=1), axis=1)
        finite = np.isfinite(values).all(axis=(1, 2)) & np.isfinite(power)
        raw_powers.extend(power[finite].tolist())
        frames.append(values.astype(np.float32))
        frame_powers.append(power)
        mod_ids.extend([sim.MOD_ID[modulation]] * len(values))
        snrs.extend([int(snr)] * len(values))
        seen_modulations.add(modulation)
        seen_snrs.add(int(snr))

    print(
        f"Input contains {len(data)} keys, {len(seen_modulations)} modulations, "
        f"and {len(seen_snrs)} SNR levels."
    )
    if not frames:
        raise ValueError("RadioML input contains no signal frames")

    X = np.concatenate(frames)
    mod = np.asarray(mod_ids, dtype=np.int16)
    snr = np.asarray(snrs, dtype=np.int16)
    raw_power = np.concatenate(frame_powers)
    finite = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(raw_power)
    keep = finite & (raw_power > 0)
    if not np.any(keep):
        raise ValueError("RadioML input contains no finite, non-zero-power frames")

    # Stats are from finite frames before normalization, including zero-power frames.
    raw_power_stats = np.asarray(raw_powers, dtype=np.float64)
    info = {
        "input_file": os.path.basename(input_path),
        "keys": len(data),
        "modulations": sorted(seen_modulations),
        "snr_levels_db": sorted(seen_snrs),
        "frames_before_cleaning": int(len(X)),
        "frames_used_for_raw_power_stats": int(len(raw_power_stats)),
        "raw_average_power_per_frame": {
            "mean": float(np.mean(raw_power_stats)),
            "min": float(np.min(raw_power_stats)),
            "max": float(np.max(raw_power_stats)),
        },
        "dropped_non_finite": int((~finite).sum()),
        "dropped_zero_power": int((finite & (raw_power <= 0)).sum()),
        "frames_kept": int(keep.sum()),
    }

    X, mod, snr, raw_power = X[keep], mod[keep], snr[keep], raw_power[keep]
    X /= np.sqrt(raw_power).astype(np.float32)[:, None, None]

    rng = np.random.default_rng(2026)
    strata = (mod.astype(np.int64) + 1) * 1000 + (snr.astype(np.int64) + 20)
    split = stratified_split(strata, rng, fr=(0.70, 0.15, 0.15))
    info["frames_per_split"] = {
        name: int(np.count_nonzero(split == index))
        for index, name in enumerate(("train", "val", "test"))
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, "radioml_2016a.npz"),
        X=X.astype(np.float32, copy=False),
        mod=mod,
        snr=snr,
        split=split.astype(np.int8, copy=False),
    )
    with open(os.path.join(OUT_DIR, "radioml_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Split sizes: {info['frames_per_split']}")
    print(f"Saved data/radioml_2016a.npz and data/radioml_info.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to RML2016.10a_dict.pkl")
    convert(parser.parse_args().input)


if __name__ == "__main__":
    main()
