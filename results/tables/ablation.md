| Configuration | Best LR | Val acc after epoch 1 (%) | Final train acc (%) | Final val acc (%) | Δ val vs original (pp) |
|---|---|---|---|---|---|
| Sigmoid + MSE, N(0,0.01²) init (original) | 3.0 | 90.76 | 98.50 ± 0.06 | 96.73 ± 0.05 | +0.00 |
| Sigmoid + softmax/CE, N(0,0.01²) init | 1.0 | 92.60 | 99.78 ± 0.02 | 97.09 ± 0.04 | +0.36 |
| Sigmoid + softmax/CE, Xavier init | 1.0 | 92.84 | 99.81 ± 0.02 | 97.08 ± 0.10 | +0.35 |
| ReLU + softmax/CE, N(0,0.01²) init | 0.3 | 93.15 | 99.83 ± 0.03 | 97.15 ± 0.12 | +0.43 |
| ReLU + softmax/CE, He init | 0.3 | 93.65 | 99.91 ± 0.02 | 97.11 ± 0.13 | +0.39 |
