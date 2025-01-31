**This is a program to train a convulutional neural network for acoustic recordings.** The program was developed specifically for amphibians within Banff National Park. The main program is designed to conduct a grid search across all of the user-defined hyperparameter values. Acoustic signal processing and Fourier transform parameters can be included in the gridsearch. Users are also able to specify the number of cross-validation folds to use during hyperparameter optimization. Training and validation results are logged through the Weights & Biases platform. Training and validation logging occurs following every epoch. A running log of the top scoring validation metric is additionally recorded based on the validation loss score. 
<br>
<br>

---
Overview of training process:
<br>
<br>
The program implements best-practices in the training procedure. Parameters at each step can be customized and/or included in the hyperparameter optimization. 
<br>
![](https://github.com/hurdg/Train-Bioacoustic-Neural-Net/blob/main/images/ProcessFlowchart.png)
<br>
<br>  

Training iterations can be monitored in real-time through the weights and biases platform.
<br>
![](https://github.com/hurdg/Train-Bioacoustic-Neural-Net/blob/main/images/wandb_snip.png)
