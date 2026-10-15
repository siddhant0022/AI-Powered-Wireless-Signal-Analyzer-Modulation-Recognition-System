# Stage A results

## Test accuracy

| Dataset | Overall test accuracy | SNR >= 10 dB | SNR < 0 dB | Test frames |
|---|---:|---:|---:|---:|
| RadioML | 0.6228 | 0.9287 | 0.3239 | 33000 |
| Simulated | 0.5785 | 0.8424 | 0.3300 | 6600 |

## Accuracy by SNR

| SNR (dB) | RadioML | Simulated |
|---:|---:|---:|
| -20 | 0.0921 | 0.1152 |
| -18 | 0.0885 | 0.0970 |
| -16 | 0.0939 | 0.1212 |
| -14 | 0.1127 | 0.2061 |
| -12 | 0.1661 | 0.2697 |
| -10 | 0.2606 | 0.3030 |
| -8 | 0.3855 | 0.3758 |
| -6 | 0.5485 | 0.4939 |
| -4 | 0.6745 | 0.5879 |
| -2 | 0.8170 | 0.7303 |
| 0 | 0.8982 | 0.7606 |
| 2 | 0.9121 | 0.8000 |
| 4 | 0.9194 | 0.8333 |
| 6 | 0.9139 | 0.8485 |
| 8 | 0.9285 | 0.8152 |
| 10 | 0.9309 | 0.8455 |
| 12 | 0.9333 | 0.8424 |
| 14 | 0.9279 | 0.8364 |
| 16 | 0.9303 | 0.8455 |
| 18 | 0.9212 | 0.8424 |

## Training details

| Dataset | Epochs trained | Best epoch | Parameters | Device | Training time (s) | Test frames |
|---|---:|---:|---:|---|---:|---:|
| RadioML | 20 | 20 | 220427 | cpu | 8302.5 | 33000 |
| Simulated | 20 | 16 | 220427 | cpu | 1491.1 | 6600 |

## Most confused class pairs at SNR >= 10 dB

### RadioML

| Class pair | Two-way confusions |
|---|---:|
| WBFM ↔ AM-DSB | 410 |
| QAM16 ↔ QAM64 | 71 |
| AM-DSB ↔ AM-SSB | 20 |

### Simulated

| Class pair | Two-way confusions |
|---|---:|
| QAM16 ↔ QAM64 | 141 |
| QPSK ↔ 8PSK | 92 |
| QAM16 ↔ PAM4 | 6 |
