# Anomaly evaluation: smoke_all

- Test frames: 7659 (6600 normal, 1059 anomaly)
- Held-out anomaly type: none
- Tables use the validation-selected maximum-F1 threshold `0.68765044`.
- Validation 95%-recall threshold: `0.1902511` (saved but not used in the tables).

## Overall test performance

| Threshold selection | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Normal FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation max F1 (used) | 0.68765044 | 0.6158 | 0.5146 | 0.5607 | 0.8699 | 0.6631 | 0.0515 |
| Validation 95% recall | 0.1902511 | 0.2136 | 0.9585 | 0.3494 | 0.8699 | 0.6631 | 0.5661 |

## Per-type detection (maximum-F1 threshold)

| Type | Exposure | Detection rate | ROC-AUC vs normal | Anomaly frames |
|---|---|---:|---:|---:|
| jamming_tone | seen | 0.5980 | 0.8709 | 204 |
| clipping | seen | 0.3184 | 0.7231 | 201 |
| impulsive_noise | seen | 0.7811 | 0.9516 | 201 |
| unknown_ofdm | seen | 0.3684 | 0.8930 | 228 |
| unknown_chirp | seen | 0.5244 | 0.9040 | 225 |

## Detection rate and normal false-positive rate by pre-fault SNR

| SNR range | Type | Detection rate | Anomaly frames | Normal FPR | Normal frames |
|---|---|---:|---:|---:|---:|
| SNR < 0 dB | jamming_tone | 0.4040 | 99 | 0.1003 | 3300 |
| SNR < 0 dB | clipping | 0.5888 | 107 | 0.1003 | 3300 |
| SNR < 0 dB | impulsive_noise | 0.5784 | 102 | 0.1003 | 3300 |
| SNR < 0 dB | unknown_ofdm | 0.2150 | 107 | 0.1003 | 3300 |
| SNR < 0 dB | unknown_chirp | 0.1604 | 106 | 0.1003 | 3300 |
| SNR 0-9 dB | jamming_tone | 0.7347 | 49 | 0.0024 | 1650 |
| SNR 0-9 dB | clipping | 0.0222 | 45 | 0.0024 | 1650 |
| SNR 0-9 dB | impulsive_noise | 1.0000 | 49 | 0.0024 | 1650 |
| SNR 0-9 dB | unknown_ofdm | 0.3333 | 57 | 0.0024 | 1650 |
| SNR 0-9 dB | unknown_chirp | 0.7759 | 58 | 0.0024 | 1650 |
| SNR >= 10 dB | jamming_tone | 0.8214 | 56 | 0.0030 | 1650 |
| SNR >= 10 dB | clipping | 0.0000 | 49 | 0.0030 | 1650 |
| SNR >= 10 dB | impulsive_noise | 0.9800 | 50 | 0.0030 | 1650 |
| SNR >= 10 dB | unknown_ofdm | 0.6562 | 64 | 0.0030 | 1650 |
| SNR >= 10 dB | unknown_chirp | 0.9180 | 61 | 0.0030 | 1650 |

## Anomaly type classification (anomaly frames only)

| True / predicted | jamming_tone | clipping | impulsive_noise | unknown_ofdm | unknown_chirp |
|---|---:|---:|---:|---:|---:|
| jamming_tone | 157 | 10 | 3 | 33 | 1 |
| clipping | 22 | 173 | 0 | 6 | 0 |
| impulsive_noise | 8 | 4 | 163 | 26 | 0 |
| unknown_ofdm | 22 | 19 | 12 | 173 | 2 |
| unknown_chirp | 18 | 12 | 4 | 56 | 135 |

| Type | Accuracy |
|---|---:|
| jamming_tone | 0.7696 |
| clipping | 0.8607 |
| impulsive_noise | 0.8109 |
| unknown_ofdm | 0.7588 |
| unknown_chirp | 0.6000 |
