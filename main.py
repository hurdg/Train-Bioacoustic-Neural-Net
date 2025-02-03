import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

import wandb
import os

from utils import get_count, get_TrainData, get_dataLoader, reset_wandb_env, get_balancedDF
from train import train


# Define sweep configuration
sweep_configuration = {
    "name": "weto",
    "method": "grid",
    #"metric": {"goal": "maximize", "name": "validation.f1"},
    "parameters": {

                #Spectrogram Creation
                'window_type' : {"values": ['hann']},
                'window_length_sec' :{"values": [None]},
                'window_samples' : {"values": [256]}, 
                'overlap_samples' : {"values": [None]},
                'overlap_fraction' : {"values": [0.5]},
                'fft_size' : {"values": [None]},

                #Data Augmentation
                'sample_rate' : {"values": [16000]},
                'random_trim' : {"values": [True]},
                'bandpass_bypass' : {"values": [True]}, #must remain off for 
                'time_mask_bypass' : {"values": [False]},
                'frequency_mask_bypass' : {"values": [False]},
                'add_noise_bypass' : {"values": [True, False]},
                'rescale_bypass' : {"values": [True, False]}, 
                'random_affine_bypass' : {"values": [True]},

                #Hyperparameters
                'dropout'  : {"values": [False]},
                'weight_decay'  : {"values": [1e-2]},
                #'learning_rate' : {"values": [1e-4, 1e-5]},
                'learning_rate' : {"values": [1e-3, 1e-4, 1e-5]},
                'stop_patience'  : {"values": [10]},
                'stop_delta'  : {"values": [0]},

                #'lr' : {"values": [0.005, 0.01, 0.02]}, # Default = 0.01
                #'lr_cooling_factor' : {"values": [0.035, 0.7, 0.14]}, # Default = 0.7
                #'lr_update_interval' : {"values": [5,10,20]}, #in epochs  Default = 10
                #'momentum' : {"values": [0.9]}, #leaky  Default = 0.9
                #'weight_decay' : {"values": [0.00025, 0.0005, 0.001]}, #l2 regularization Default = 0.0005
                
                'architecture' :{"values": ['resnet34']}, 
                'height' :{"values": [224]}, 
                'width' :{"values": [224]},
                "n_balance":{"values":[1000]},
                "epochs":{"values":[100]},
                "batch_size":{"values":[32]}, #must keep low when using dropout(inplace=False)!!!
    },
}

def cross_validate():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#--Sweep environment
    #Set name of run
    count = get_count()
    sweep_run_name = " ".join(['weto','run', str(count)])

    sweep_run = wandb.init()
    sweep_run.name = sweep_run_name
    sweep_id = sweep_run.sweep_id
    sweep_run_id = sweep_run.id
    config = sweep_run.config
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)
#--

    val_metrics = {'val_acc':[], 'val_prc':[], 'val_rec':[], 'val_loss':[]}
    train_metrics = {'train_acc':[], 'train_prc':[], 'train_rec':[], 'train_loss': []}

    #Get all training data
    labeled_df = get_TrainData(data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data2'))

    #Split for cross validation, stratify with respect to presence
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(labeled_df, labeled_df['presence'], labeled_df['group']), 1):
        train_df = get_balancedDF(labeled_df.iloc[train_idx], balance=True, n_balance = config['n_balance'], single_class=True, type = 'train')
        valid_df = get_balancedDF(labeled_df.iloc[val_idx], balance=False, n_balance = config['n_balance'], single_class=True, type = 'validation')
        reset_wandb_env()
        train_dataloader = get_dataLoader(train_df, config, train=True)
        valid_dataloader = get_dataLoader(valid_df, config, train=False)

        val_dict, train_dict = train(   train_dataloader, 
                                        valid_dataloader, 
                                        config,
                                        sweep_id, sweep_run_name, fold)
        for key, value in val_dict.items():
            val_metrics[key].append(value)

        for key, value in train_dict.items():
            train_metrics[key].append(value)

    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(dict(val_accuracy=sum(val_metrics['val_acc']) / len(val_metrics['val_acc']),
                       val_precision=sum(val_metrics['val_prc']) / len(val_metrics['val_prc']),
                       val_recall=sum(val_metrics['val_rec']) / len(val_metrics['val_rec']),
                       val_loss=sum(val_metrics['val_loss']) / len(val_metrics['val_loss']),
                       val_specificity=sum(val_metrics['val_spc']) / len(val_metrics['val_spc']),
                       val_macro_accuracy=sum(val_metrics['val_macro_acc']) / len(val_metrics['val_macro_acc']),
                       train_accuracy=sum(train_metrics['train_acc']) / len(train_metrics['train_acc']),
                       train_precision=sum(train_metrics['train_prc']) / len(train_metrics['train_prc']),
                       train_recall=sum(train_metrics['train_rec']) / len(train_metrics['train_rec']),
                       train_loss=sum(train_metrics['train_loss']) / len(train_metrics['train_loss'])))
    sweep_run.finish()



def main():
    import warnings
    warnings.filterwarnings('ignore')

    # Define counter variable from which to begin sequential run naming (count will begin at specified 'counter' value +1)
    # Leave at 0, unless restarting a crashed/terminated sweep    
    counter=0

    # Decision to continue previous sweep (True = Continue)
    rerun = False
    if not rerun:

        #Initiate a new sweep with specified parameters (above)
        wandb.login()
        sweep_id = wandb.sweep(sweep_configuration, project='weto')
        wandb.agent(sweep_id, function=cross_validate)

    else:
        #Re-initiate a previously commenced sweep with the same sweep_id 
        wandb.agent(sweep_id='dngkjap7', project="weto_sweep", function=cross_validate)

    #Terminate WANDB session
    wandb.finish()


if __name__ == "__main__":
    main()