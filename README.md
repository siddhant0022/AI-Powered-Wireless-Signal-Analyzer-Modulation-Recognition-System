# AI-Powered Wireless Signal Analyzer — October Deliverable: Ready-to-Use Dataset

**Plan item:** *Collect and simulate signals with GNU Radio. Clean and label the data.*  →  **Deliverable: ready-to-use dataset.**
Team A13 · ECE-ACT · Galgotias College of Engineering and Technology · Guide: Dr. Deepika Rajpoot

## 1. Quick start
```bash
pip install -r requirements.txt
python build_dataset.py            # generate + clean + label + split   (~20 s)
python validate_and_plot.py        # 12 integrity checks + statistics + plots
python dataset_loader.py           # prints split sizes
```
```python
from dataset_loader import load_dataset
train = load_dataset("train")      # "val", "test", "all"
X, y_mod, y_snr = train["X"], train["mod"], train["snr"]      # X: (N, 2, 128)
normal = load_dataset("train", normal_only=True)              # for anomaly-detector training
```
The generated dataset is already included in `data/`, so you can use it without rebuilding.

## 2. What is in the dataset
| Item | Value |
|---|---|
| Frame | 128 complex samples stored as float32 array `(2, 128)` (row 0 = I, row 1 = Q), unit average power |
| Modulations (11) | BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, GFSK, CPFSK, WBFM, AM-DSB, AM-SSB (same classes as RadioML 2016.10A) |
| SNR levels (20) | −20 to +18 dB, step 2 dB |
| Channels (2) | `awgn` (frequency + phase offset) and `fading` (3-tap Rayleigh multipath) |
| Normal frames | 44,000  (11 mod × 20 SNR × 2 channels × 100) |
| Anomaly frames | 7,500  (5 types × 1,500) |
| **Total** | **51,500 frames** (≈ 49 MB) |
| Split | stratified train 36,100 / val 7,741 / test 7,659 (≈ 70/15/15) |

### Labels (every frame)
`mod` (modulation id, −1 = unknown), `snr` (dB, regression target), `quality` (Poor <0 dB, Fair 0–9 dB, Good ≥10 dB),
`channel`, `anomaly` (0/1), `anomaly_type`, `split`.

### Anomaly types
| Type | How it is made | Why it matters |
|---|---|---|
| `jamming_tone` | strong CW tone added, J/S 0–10 dB | interference / jamming |
| `clipping` | I and Q hard-clipped at 25–60 % of peak | amplifier saturation |
| `impulsive_noise` | sparse high-power bursts | switching / lightning noise |
| `unknown_ofdm` | OFDM-like multicarrier signal | modulation not in the 11 known classes |
| `unknown_chirp` | linear FM sweep | modulation not in the 11 known classes |

## 3. Pipeline (matches the plan)
1. **Simulate** – `simulator.py`: source → modulator → RRC pulse shaping (β=0.35, 8 samples/symbol) → channel → AWGN.
   Same chain as GNU Radio's RadioML dataset generator (O'Shea & West, 2016).
2. **Clean** – `build_dataset.py`: remove NaN/Inf and zero-energy frames (logged in `dataset_info.json`), normalise each frame to unit power.
3. **Label** – modulation, SNR, quality, channel, anomaly flag and type for every frame (`labels.csv` is human-readable).
4. **Split** – stratified by modulation × SNR × anomaly type, so every split has all classes at all SNRs.
5. **Validate** – `validate_and_plot.py`: 12 automatic checks (all pass), statistics JSON and 5 plots in `plots/`.

## 4. Files
| File | Purpose |
|---|---|
| `simulator.py` | signal generators, channel, anomaly injection |
| `build_dataset.py` | build, clean, label, split, save |
| `dataset_loader.py` | `load_dataset()` helper |
| `validate_and_plot.py` | integrity checks, statistics, plots |
| `import_radioml.py` | optional: convert real RadioML 2016.10A into the same format |
| `data/signal_dataset.npz` | the dataset |
| `data/labels.csv` | readable label table |
| `data/dataset_info.json`, `dataset_statistics.json` | metadata and counts |
| `plots/` | distributions, modulation gallery, SNR effect, anomaly examples, feature sanity check |

## 5. Dataset card — conventions and limitations (be upfront about these in the review)
* **SNR definition:** signal power / noise power over the full sampled bandwidth (the RadioML convention). Signals are scaled to unit power before noise is added.
* **Anomaly frames:** `snr` is the SNR *before* the anomaly was injected; for unknown signals it is the SNR of that signal.
* **Simulation tool:** the signals are generated with NumPy/SciPy, replicating the GNU Radio chain, so the dataset is reproducible with no GNU Radio installation. (A GNU Radio flowgraph can be used to generate additional files in the same format.)
* **Analog sources** (WBFM, AM-DSB, AM-SSB) use a band-limited random message, not real speech/music.
* **Synthetic data:** channel effects are simplified (no hardware impairments). Real-signal validation uses RadioML via `import_radioml.py` (download needed, not included).
* **Reproducibility:** fixed seed (2026); `python build_dataset.py --seed N --per-cell M` regenerates or enlarges it.
