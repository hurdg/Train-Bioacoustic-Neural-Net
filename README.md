This is a program to train a convulutional neural network for acoustic recordings. The program was developed specifically for amphibians within Banff National Park. The main program is designed to conduct a grid search across all of the user-defined hyperparameter values. Acoustic signal processing and Fourier transform parameters can be included in the gridsearch. Users are also able to specify the number of cross-validation folds to use during hyperparameter optimization. Training and validation results are logged through the Weights & Biases platform. Training and validation logging occurs following every epoch. A running log of the top scoring validation metric is additionally recorded based on the validation loss score. 
<br>
<br>
---
Overview of user interface:
<br>
<br>
The interface contains features that facilitate navigation through audio recordings. 
<br>
![]((https://github.com/hurdg/Train-Bioacoustic-Neural-Net/blob/main/images/ProcessFlowchart.png)) 
<br>
<br>
The interface also supports both manual and automated classification. Automated classification is based on the neural networks predictions in relation a user-defined threshold value. An upper and lower threshold value can be specified. In the case of a conflict between the manual and automated classifications, priority will be given to the former.
<br>
![](https://github.com/hurdg/amphibian-bioacoustics-user-interface/blob/main/images/UI_annotation2.png)
