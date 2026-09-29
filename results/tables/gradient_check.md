| Hidden act. | Loss | Init | Layers | Max rel. error | Result |
|---|---|---|---|---|---|
| sigmoid | sigmoid_mse | normal_0.01 | 20-10-10 | 1.8e-07 | pass |
| sigmoid | softmax_ce | xavier | 20-10-10 | 3.2e-08 | pass |
| relu | softmax_ce | he | 20-10-10 | 2.6e-07 | pass |
| relu | softmax_ce | he | 20-12-8-10 | 4.9e-08 | pass |
