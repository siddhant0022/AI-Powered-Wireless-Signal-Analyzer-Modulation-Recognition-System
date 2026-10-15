# Simulated Stage A: original vs narrower CFO

Both rows are computed from the saved best checkpoints and their own normal, known-modulation test split.

## Test accuracy by SNR

| SNR (dB) | Original simulated (accuracy; correct / n) | CFO max 0.001 (accuracy; correct / n) |
|---:|---:|---:|
| -20 | 11.5% (38/330) | 13.3% (44/330) |
| -18 | 9.7% (32/330) | 12.4% (41/330) |
| -16 | 12.1% (40/330) | 14.5% (48/330) |
| -14 | 20.6% (68/330) | 24.8% (82/330) |
| -12 | 27.0% (89/330) | 25.5% (84/330) |
| -10 | 30.3% (100/330) | 28.5% (94/330) |
| -8 | 37.6% (124/330) | 38.5% (127/330) |
| -6 | 49.4% (163/330) | 52.1% (172/330) |
| -4 | 58.8% (194/330) | 61.2% (202/330) |
| -2 | 73.0% (241/330) | 70.6% (233/330) |
| 0 | 76.1% (251/330) | 80.3% (265/330) |
| 2 | 80.0% (264/330) | 84.8% (280/330) |
| 4 | 83.3% (275/330) | 86.7% (286/330) |
| 6 | 84.8% (280/330) | 89.7% (296/330) |
| 8 | 81.5% (269/330) | 87.9% (290/330) |
| 10 | 84.5% (279/330) | 90.3% (298/330) |
| 12 | 84.2% (278/330) | 90.9% (300/330) |
| 14 | 83.6% (276/330) | 87.6% (289/330) |
| 16 | 84.5% (279/330) | 90.0% (297/330) |
| 18 | 84.2% (278/330) | 89.1% (294/330) |

## Selected confusions at SNR >= 10 dB

Pair counts combine both confusion directions. Each run uses all normal, known-modulation test frames at SNR >= 10 dB (both channels).

| Class pair | Original simulated | CFO max 0.001 |
|---|---:|---:|
| QPSK ↔ 8PSK | 92 | 12 |
| QAM16 ↔ QAM64 | 141 | 139 |

| Test frames at SNR >= 10 dB | Original simulated | CFO max 0.001 |
|---|---:|---:|
| Count | 1650 | 1650 |

Accuracy plot: `simulated_cfo0001_accuracy_vs_snr.png`.
