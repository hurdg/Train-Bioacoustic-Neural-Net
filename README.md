# Bioacoustic Neural Network Training

**This program trains a convolutional neural network (CNN) for acoustic recordings.** It was developed specifically for amphibians within **Banff National Park**. The program is designed to conduct a grid search across all user-defined hyperparameter values, including acoustic signal processing and Fourier transform parameters.

Users can specify the number of cross-validation folds to use during hyperparameter optimization. Training and validation results are logged using the **Weights & Biases** platform, with logs updated after every epoch. Additionally, a running log of the top scoring validation metric is recorded based on the validation loss score.

---

## Overview of the Training Process

The program follows best practices in the training procedure, and parameters at each step can be customized and included in the hyperparameter optimization.

![Training Process Flowchart](https://github.com/hurdg/Train-Bioacoustic-Neural-Net/blob/main/images/ProcessFlowchart.png)

Training iterations can be monitored in real-time through the **Weights & Biases** platform.

![Weights & Biases Monitoring](https://github.com/hurdg/Train-Bioacoustic-Neural-Net/blob/main/images/wandb_snip.png)

---
