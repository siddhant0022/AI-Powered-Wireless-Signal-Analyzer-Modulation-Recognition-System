# AI-Powered Wireless Signal Analyzer

This project analyzes short wireless I/Q signal frames and predicts their modulation type.

It contains:

- a NumPy/SciPy wireless signal simulator;
- an I/Q dataset builder with labels and train/validation/test splits;
- a converter for the RadioML 2016.10A dataset;
- a PyTorch 1D convolutional neural network;
- training, evaluation, visualization, comparison, and test scripts.

The completed machine-learning stage is **Stage A: modulation recognition**. It classifies a 128-sample I/Q frame into one of 11 modulation classes.

## Simple overview

**What does the project receive?**

Each signal frame is represented by 128 complex samples. The project stores these samples as a real array with shape `(2, 128)`:

- row 0: in-phase component (I);
- row 1: quadrature component (Q).

**What does the project predict?**

One of these 11 modulations:

`BPSK`, `QPSK`, `8PSK`, `QAM16`, `QAM64`, `PAM4`, `GFSK`, `CPFSK`, `WBFM`, `AM-DSB`, `AM-SSB`

**What has been done?**

1. Generated and labeled a simulated wireless dataset.
2. Converted and normalized the RadioML 2016.10A dataset.
3. Added cleaning, validation, metadata, and split logic.
4. Built and trained a 1D CNN.
5. Evaluated the model on each SNR level and as an overall test set.
6. Compared the simulated and RadioML datasets.
7. Tested a lower-carrier-frequency-offset experiment.
8. Saved trained checkpoints, metrics, graphs, and reports.

**What is not implemented yet?**

Anomaly detection is prepared for future work. The default Stage A pipeline:

- anomaly frames are excluded from modulation training;
- only the 11-class modulation head is used.

An optional joint modulation-and-SNR training mode is available with `--use-snr`.

## End-to-end workflow

```text
Signal source
  - Python simulator
  - RadioML 2016.10A pickle

          |
          v

Dataset preparation
  - Generate or import frames
  - Remove invalid frames
  - Normalize I/Q power
  - Add modulation, SNR, channel, anomaly, and quality labels
  - Split into train / validation / test

          |
          v

Data checking
  - Verify shapes and values
  - Check class balance
  - Plot distributions and example signals

          |
          v

Training
  - Load known, normal modulation frames
  - Apply random phase rotation
  - Train the CNN
  - Default Stage A: select by validation cross-entropy, then evaluate the test set
  - Joint SNR mode: select by combined validation modulation accuracy and SNR error

          |
          v

Results
  - Overall accuracy
  - Accuracy at every SNR
  - Accuracy for low and high SNR groups
  - Confusion matrix
  - Checkpoint and training history
```

## Project structure

```text
.
├── simulator.py                 Generate modulation frames and channel effects
├── build_dataset.py             Build, clean, label, and split simulated data
├── import_radioml.py            Convert RadioML 2016.10A to NPZ
├── dataset_loader.py            Simple interface for loading datasets
├── validate_and_plot.py         Validate data and create dataset plots
├── compare_datasets.py          Compare RadioML and simulated signals
├── model.py                     1D CNN and optional SNR head
├── train.py                     Train and evaluate Stage A
├── compare_results.py           Compare saved Stage A runs
├── analyze_extra.py             Create extra low-SNR and class reports
├── analyze_extra2.py            Create prediction-distribution reports
├── compare_cfo_results.py       Compare normal and lower-CFO simulated runs
├── tests/test_stageA.py         Model and split tests
├── data/                        Datasets, labels, metadata, and comparisons
├── plots/                       Dataset and signal-analysis graphs
└── results/                     Checkpoints, metrics, reports, and graphs
```

## Dataset details

### Simulated dataset

The default simulated dataset is `data/signal_dataset.npz`.

It contains:

- 51,500 total frames;
- 44,000 normal modulation frames;
- 7,500 anomaly frames;
- 11 modulations;
- 20 SNR levels from -20 dB through +18 dB;
- AWGN and 3-tap Rayleigh fading channels;
- a 70/15/15 train/validation/test split.

The normal frames are balanced across 11 modulations, 20 SNR values, and 2 channels.

Anomaly types are:

- `jamming_tone`
- `clipping`
- `impulsive_noise`
- `unknown_ofdm`
- `unknown_chirp`

### RadioML dataset

`data/radioml_2016a.npz` is a normalized copy of RadioML 2016.10A in the format used by this project.

It contains:

- 220,000 frames;
- 154,000 training frames;
- 33,000 validation frames;
- 33,000 test frames;
- the same 11 modulations and SNR range used in the project.

RadioML's original channel and anomaly labels are not present. The project loader therefore marks RadioML channel data as unavailable and all frames as non-anomaly.

### Dataset fields

| Field | Type | Meaning |
|---|---|---|
| `X` | float32 array | I/Q frames with shape `(N, 2, 128)` |
| `mod` | int16 array | Modulation ID, 0 to 10; `-1` means unknown |
| `snr` | int16 array | SNR in dB |
| `split` | int8 array | `0` train, `1` validation, `2` test |
| `channel` | int8 array | `0` AWGN, `1` fading, `-1` unavailable |
| `anomaly` | int8 array | `0` normal, `1` anomaly |
| `anomaly_type` | int8 array | None or one of five anomaly types |
| `quality` | int8 array | `0` poor, `1` fair, `2` good |

Quality is derived from SNR:

- poor: SNR below 0 dB;
- fair: SNR from 0 dB through 9 dB;
- good: SNR of 10 dB or above.

## How the CNN works

The Stage A model is a compact 1D CNN in `model.py`.

It:

1. receives I/Q data shaped `(batch, 2, 128)`;
2. applies 1D convolution and residual blocks;
3. combines average and maximum pooled features;
4. passes the features through a dense layer;
5. produces 11 modulation logits.

The model has approximately **220,427 parameters**.

During training, the I/Q frame is randomly rotated in the complex plane. This is phase augmentation and does not change signal power.

Training uses:

- Adam optimizer;
- cross-entropy loss;
- validation loss for best-checkpoint selection;
- early stopping;
- a test split that is evaluated only after training ends.

## Stage A results

These are the included recorded runs. A new run may produce different values because of dataset generation, environment, or training settings.

| Dataset | Overall test accuracy | SNR >= 10 dB | SNR < 0 dB | Test frames |
|---|---:|---:|---:|---:|
| RadioML | 62.28% | 92.87% | 32.39% | 33,000 |
| Simulated | 57.85% | 84.24% | 33.00% | 6,600 |
| Simulated, CFO max 0.001 | 60.94% | 89.58% | 34.15% | 6,600 |

The main reports are:

- [Stage A summary](results/stageA_summary.md)
- [Low-CFO comparison](results/simulated_cfo0001_comparison.md)
- [Low-SNR per-class report](results/stageA_per_class_low_snr.md)
- [Prediction-distribution report](results/stageA_prediction_distribution.md)
- [Simulated channel report](results/stageA_simulated_by_channel.md)

![Test accuracy versus SNR](results/stageA_both_accuracy_vs_snr.png)

Accuracy is generally highest at high SNR and lowest at strongly negative SNR.

## Setup

These steps are for Windows PowerShell and assume Git and Python are installed.

```powershell
cd C:\path\to\AI-Powered-Wireless-Signal-Analyzer-Modulation-Recognition-System

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies:

- NumPy
- SciPy
- Matplotlib
- PyTorch
- pytest

A CUDA GPU is optional. PyTorch uses it automatically when available and otherwise uses CPU.

If PowerShell does not allow environment activation, use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe validate_and_plot.py
```

The large RadioML source pickle and converted RadioML dataset are kept out of GitHub because each exceeds GitHub's regular-file size limit. They remain on the current machine. To create the simulated dataset, run `python build_dataset.py`. To use RadioML, place `RML2016.10a_dict.pkl` in `data/` and run the importer described below. Generated reports and results remain available in the repository.

## Quick start

Check the included simulated dataset:

```powershell
python validate_and_plot.py
python dataset_loader.py
```

`validate_and_plot.py` writes validation statistics and plots. `dataset_loader.py` prints split and anomaly counts.

Run the tests:

```powershell
pytest -q
```

## Loading data in Python

```python
from dataset_loader import load_dataset

# Simulated training data
train = load_dataset("train")

X = train["X"]            # (N, 2, 128)
mod = train["mod"]        # modulation IDs
snr = train["snr"]        # SNR in dB

# RadioML test data
radio_test = load_dataset("test", dataset="radioml")

# Simulated normal frames only
normal_train = load_dataset("train", normal_only=True)

# Simulated anomaly frames only
anomalies = load_dataset("all", anomalies_only=True)
```

Supported split names are `train`, `val`, `test`, and `all`.

## Rebuild the simulated dataset

The normal command creates the default dataset:

```powershell
python build_dataset.py
```

Useful options:

```powershell
python build_dataset.py --per-cell 100
python build_dataset.py --per-cell 500 --seed 7
python build_dataset.py --cfo-max 0.001 --seed 2026
```

The default output files are:

- `data/signal_dataset.npz`
- `data/labels.csv`
- `data/dataset_info.json`

A CFO value other than `0.01` creates separate `signal_dataset_cfo...` files so the default dataset is not overwritten.

## Import RadioML

Place the original `RML2016.10a_dict.pkl` in `data/` or use its path:

```powershell
python import_radioml.py data\RML2016.10a_dict.pkl
```

This writes:

- `data/radioml_2016a.npz`
- `data/radioml_info.json`

## Compare datasets

```powershell
python compare_datasets.py
```

This writes `data/comparison.json` and refreshes plots in `plots/`.

## Train Stage A

The trainer defaults to the converted RadioML dataset.

Quick simulated smoke test:

```powershell
python train.py --data data\signal_dataset.npz --epochs 1 --seed 2026 --tag smoke
```

Full simulated run:

```powershell
python train.py --data data\signal_dataset.npz --epochs 20 --seed 2026 --tag simulated_stageA
```

Full RadioML run:

```powershell
python train.py --data data\radioml_2016a.npz --epochs 20 --seed 2026 --tag radioml_stageA
```

Joint modulation and SNR training:

```powershell
python train.py --dataset simulated --use-snr --epochs 60 --run-name simulated_snr
python train.py --dataset radioml --use-snr --epochs 60 --run-name radioml_snr
```

The joint model selects checkpoints using
`val_mod_acc - 0.5 * val_snr_mae_db / 20`. Use `--resume` with the same run name
to continue from its `last.pt`; the requested `--epochs` is the final epoch number.
The SNR regression loss uses SmoothL1 with `--snr-loss-beta` (default `1.0`).
Runs save `best.pt`, `last.pt`, `history.csv`, and `config.json` under
`results/<run-name>/`. The default modulation-only path retains validation
cross-entropy checkpoint selection and its Stage A test evaluation.

Joint modulation, SNR, and anomaly training uses only the simulated dataset:

```powershell
python train.py --dataset simulated --use-anomaly --use-anomaly-type --epochs 60 --run-name simulated_all_heads
```

The anomaly checkpoint score is
`val_mod_acc - 0.5 * val_snr_mae_db / 20 + val_anomaly_f1`. To hold an anomaly
type out of training and validation for a later unseen-type evaluation, use:

```powershell
python train.py --dataset simulated --use-anomaly --use-anomaly-type --holdout-anomaly-type unknown_ofdm --epochs 60 --run-name holdout_unknown_ofdm
python train.py --dataset simulated --use-anomaly --use-anomaly-type --holdout-anomaly-type unknown_chirp --epochs 60 --run-name holdout_unknown_chirp
python train.py --dataset simulated --use-anomaly --use-anomaly-type --holdout-anomaly-type jamming_tone --epochs 60 --run-name holdout_jamming_tone
```

Evaluate a joint model on the held-out test split:

```powershell
python evaluate_snr.py --run-name simulated_snr --dataset simulated
python evaluate_snr.py --run-name radioml_snr --dataset radioml
```

Evaluation reports and 300-dpi plots are written to
`results/<run-name>/eval/`. The evaluator alone loads the test split for joint
modulation/SNR runs; it also reports the change in modulation accuracy against
the recorded Stage A baseline.

Evaluate anomaly detection and aggregate the three requested holdout runs:

```powershell
python evaluate_anomaly.py --run-name simulated_all_heads
python evaluate_anomaly.py --run-name holdout_unknown_ofdm
python evaluate_anomaly.py --run-name holdout_unknown_chirp
python evaluate_anomaly.py --run-name holdout_jamming_tone
python aggregate_holdout.py
```

Anomaly reports and plots are written under
`results/<run-name>/eval_anomaly/`; aggregate seen/unseen results are written to
`results/holdout_aggregate.md`. Validation selects both the maximum-F1 threshold
and the 95%-recall threshold. Report tables use the maximum-F1 threshold.

View all trainer options:

```powershell
python train.py --help
```

Each run is saved in `results/<tag>/` and normally contains:

- `best_model.pt`: saved model weights;
- `history.json`: loss and accuracy by epoch;
- `train_log.txt`: console training output;
- `summary.json`: overall and SNR-group metrics;
- `accuracy_vs_snr.csv` and `.png`;
- `loss_curve.png`;
- `confusion_matrix_snr_ge_10.png`.

Choose a new `--tag` when creating a run that should keep separate output.

## Create reports and plots

```powershell
python validate_and_plot.py
python compare_datasets.py
python compare_results.py
python analyze_extra.py
python analyze_extra2.py
pytest -q
```

The comparison and analysis scripts use saved datasets and result files. They do not retrain the model.

## Repeat the lower-CFO experiment

The default CFO maximum is `0.01` of the sample rate. The experiment uses `0.001`.

```powershell
python build_dataset.py --cfo-max 0.001 --seed 2026

python train.py --data data\signal_dataset_cfo0001.npz --epochs 20 --seed 2026 --tag simulated_cfo0001

python compare_cfo_results.py
```

This writes:

- `data/signal_dataset_cfo0001.npz`
- `data/signal_dataset_cfo0001_info.json`
- `data/signal_dataset_cfo0001_labels.csv`
- `results/simulated_cfo0001_comparison.md`
- `results/simulated_cfo0001_accuracy_vs_snr.png`

## Common questions

### Do I need an SDR or radio hardware?

No. The included simulator creates synthetic baseband signals in NumPy. RadioML can also be used when the local pickle file is available.

### Is GNU Radio required?

No. GNU Radio is an inspiration for the signal-chain structure, but the current project is implemented in Python and SciPy.

### What is an I/Q frame?

I is the in-phase component and Q is the quadrature component of a complex wireless signal. They are stored as two 128-value rows.

### What is SNR?

SNR is signal-to-noise ratio, measured in dB. Higher values contain less relative noise.

### What is CFO?

CFO is carrier-frequency offset. It represents a frequency difference between the transmitted signal and receiver carrier. The lower-CFO experiment reduces the maximum synthetic offset.

### Why can the model perform better at high SNR?

At high SNR, noise has less effect, so modulation patterns are easier to separate.

### What is a test split?

The test split contains frames held out during training. It provides the final, unseen accuracy estimate after validation selects the checkpoint.

### Why is the test split evaluated only after training?

This prevents the test set from influencing checkpoint selection and gives a more honest final estimate.

### Does this project detect anomalies?

Not yet. Anomaly frames and labels are generated, but Stage A only trains the 11-class modulation head.

### Can the project estimate SNR?

Yes. Use `train.py --dataset simulated --use-snr` or
`train.py --dataset radioml --use-snr` to train the optional SNR head jointly
with modulation classification. The default Stage A run remains modulation-only.

### Can I change the number of training epochs?

Yes. Use `--epochs`. The default is 60, while the included recorded runs used 20 epochs.

### Can I use CPU instead of GPU?

Yes. CPU is supported and is used automatically if CUDA is unavailable.

## Limitations

- simulated analog messages are generated approximations, not real speech or music;
- the synthetic channel model is simplified;
- Stage A does not yet detect anomalies;
- the optional SNR mode does not yet produce a dedicated test-set SNR report;
- the CNN is evaluated on fixed datasets and saved seeds;
- lower-CFO results show one controlled experiment, not a guarantee for real radio hardware;
- RadioML and simulated data are different sources and should not be treated as identical test conditions.

## Possible next stages

- train and evaluate an anomaly classifier;
- add dedicated test-set SNR metrics;
- test more seeds and dataset sizes;
- add stronger channel and hardware impairments;
- evaluate on additional real-world datasets;
- provide real-time or SDR input in a future application interface.
