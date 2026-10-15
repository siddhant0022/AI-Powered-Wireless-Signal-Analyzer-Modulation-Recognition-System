# Stage A prediction distribution

Predictions were generated from the saved best checkpoints and the original test splits. No training was performed.

## RadioML: SNR -12 dB (1650 test frames)

Precision is correct predictions for the class divided by all frames predicted as that class. Counts are restricted to this SNR.

| Class | Predicted count | True count | Precision (correct / predicted) |
|---|---:|---:|---:|
| BPSK | 47 | 150 | 4.3% (2/47) |
| QPSK | 56 | 150 | 0.0% (0/56) |
| 8PSK | 24 | 150 | 0.0% (0/24) |
| QAM16 | 17 | 150 | 23.5% (4/17) |
| QAM64 | 91 | 150 | 94.5% (86/91) |
| PAM4 | 44 | 150 | 34.1% (15/44) |
| GFSK | 27 | 150 | 14.8% (4/27) |
| CPFSK | 45 | 150 | 4.4% (2/45) |
| WBFM | 10 | 150 | 30.0% (3/10) |
| AM-DSB | 75 | 150 | 22.7% (17/75) |
| AM-SSB | 1214 | 150 | 11.6% (141/1214) |

## RadioML: SNR -10 dB (1650 test frames)

Precision is correct predictions for the class divided by all frames predicted as that class. Counts are restricted to this SNR.

| Class | Predicted count | True count | Precision (correct / predicted) |
|---|---:|---:|---:|
| BPSK | 87 | 150 | 14.9% (13/87) |
| QPSK | 50 | 150 | 14.0% (7/50) |
| 8PSK | 45 | 150 | 4.4% (2/45) |
| QAM16 | 49 | 150 | 24.5% (12/49) |
| QAM64 | 131 | 150 | 74.0% (97/131) |
| PAM4 | 90 | 150 | 60.0% (54/90) |
| GFSK | 44 | 150 | 34.1% (15/44) |
| CPFSK | 48 | 150 | 8.3% (4/48) |
| WBFM | 53 | 150 | 39.6% (21/53) |
| AM-DSB | 143 | 150 | 44.1% (63/143) |
| AM-SSB | 910 | 150 | 15.6% (142/910) |

## Simulated: SNR -12 dB (330 test frames)

Precision is correct predictions for the class divided by all frames predicted as that class. Counts are restricted to this SNR.

| Class | Predicted count | True count | Precision (correct / predicted) |
|---|---:|---:|---:|
| BPSK | 22 | 30 | 13.6% (3/22) |
| QPSK | 14 | 30 | 28.6% (4/14) |
| 8PSK | 9 | 30 | 0.0% (0/9) |
| QAM16 | 6 | 30 | 33.3% (2/6) |
| QAM64 | 6 | 30 | 0.0% (0/6) |
| PAM4 | 20 | 30 | 30.0% (6/20) |
| GFSK | 40 | 30 | 22.5% (9/40) |
| CPFSK | 52 | 30 | 28.8% (15/52) |
| WBFM | 56 | 30 | 21.4% (12/56) |
| AM-DSB | 53 | 30 | 41.5% (22/53) |
| AM-SSB | 52 | 30 | 30.8% (16/52) |

## Simulated: SNR -10 dB (330 test frames)

Precision is correct predictions for the class divided by all frames predicted as that class. Counts are restricted to this SNR.

| Class | Predicted count | True count | Precision (correct / predicted) |
|---|---:|---:|---:|
| BPSK | 14 | 30 | 21.4% (3/14) |
| QPSK | 24 | 30 | 20.8% (5/24) |
| 8PSK | 6 | 30 | 33.3% (2/6) |
| QAM16 | 15 | 30 | 26.7% (4/15) |
| QAM64 | 19 | 30 | 10.5% (2/19) |
| PAM4 | 32 | 30 | 28.1% (9/32) |
| GFSK | 58 | 30 | 25.9% (15/58) |
| CPFSK | 36 | 30 | 38.9% (14/36) |
| WBFM | 33 | 30 | 27.3% (9/33) |
| AM-DSB | 42 | 30 | 45.2% (19/42) |
| AM-SSB | 51 | 30 | 35.3% (18/51) |

## Simulated AWGN confusions at SNR >= 10 dB

Test frames used: **836** (normal, known-modulation test frames on the AWGN channel with SNR >= 10 dB).

Counts combine both directions for each unordered pair.

| Rank | Class pair | Two-way confusion count |
|---:|---|---:|
| 1 | QAM16 ↔ QAM64 | 72 |
| 2 | QPSK ↔ 8PSK | 46 |
| 3 | QAM16 ↔ PAM4 | 2 |
| 4 | BPSK ↔ QPSK | 1 |
| 5 | QPSK ↔ GFSK | 1 |
