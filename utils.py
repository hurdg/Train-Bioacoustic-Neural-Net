import opensoundscape
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioFileDataset
from opensoundscape.ml.utils import collate_audio_samples_to_tensors

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn

import os
counter = 0
def get_count():

#---
# Function to count the iteration of sweep hyperparameters.
# Used to sequentially name the configurations of parameters within a WANDB sweep
#---
     ## Access global variable 'counter' (defined below)
     global counter
     # Add 1 to var
     counter = counter + 1
     return counter


def get_TrainData(data_dir, balance:bool=False, n_balance = int, single_class:bool = True):

#---
# Function to create metadata CSV for OpenSoundScape module
# Metadata csv consists of pandas dataframe:
    #Index of dataframe must be set as filepath to audiosamples
    #Columns of dataframe must be one-hot-encoded labels of species and/or positive/negative samples
#---

    #Define dict of filepaths and presence status (pos/neg) to be populated within loops
    filepath_presence_dict = {"filepath":[], "presence":[]}

    #Define path to main training data folder
    #weto_train_dir = os.path.join(data_dir, 'weto', 'train')
    weto_train_dir = os.path.join(data_dir, 'weto', 'train')

    #Iteratively loop through audio files in positive folder, then negative
    for set_key in ['positive', 'negative']:
        set_dir = os.path.join(weto_train_dir, set_key)

        #Create list of audio files in specified directory (pos/neg)
        filenames = os.listdir(set_dir)

        #Append filepaths and presence status to dict (empty, if first run through parent loop)
        filepath_list = filepath_presence_dict["filepath"]
        presence_list = filepath_presence_dict["presence"]

        #Loop through filenames to create list of filepaths and presence status
        for name in filenames:
            #Define filepath and append to list
            filepath = os.path.join(set_dir, name)
            filepath_list.append(filepath)

        #Append list to dict, as a value to 'filepath' key
        filepath_presence_dict.update({'filepath': filepath_list})

                #Create sequence of presence status values (pos/neg) of equal length to filepath list and append to dict, as a value to 'presence' key
        if set_key == "positive":
            presence_list = list(np.repeat(1, len(filenames)))
            filepath_presence_dict.update({'presence': presence_list})
        if set_key == "negative":
            presence_list.extend(list(np.repeat(0, len(filenames))))
            filepath_presence_dict.update({'presence': presence_list}) 

    samples_df = pd.DataFrame(filepath_presence_dict).set_index('filepath')

    return(samples_df)


def get_balancedDF(df, balance:bool, n_balance, single_class:bool, type:str):
    #Use get_dummies to one-hot-encode the presence column
    n_pos = len(df[df['presence']==1])
    n_neg = len(df[df['presence']==0])
    dummy_df = pd.get_dummies(df, prefix="", prefix_sep='', columns=['presence'], dtype = int)

    #Drop/repeat samples to the training set, if desired. THis is helpful to ensure balance among the classes (improves model performance)
    if balance:
        dummy_df = opensoundscape.data_selection.resample(dummy_df,n_samples_per_class=n_balance,random_state=0)
        print(f"Created balanced {type} df with {np.sum(dummy_df['1']==1)} positive samples (from {n_pos}) and {np.sum(dummy_df['0']==1)} negative samples (from {n_neg})")
    else:
        print(f"Left {type} df unbalanced with {np.sum(dummy_df['1']==1)} positive samples and {np.sum(dummy_df['0']==1)} negative samples")

    if single_class:
        dummy_df = pd.DataFrame(dummy_df['1'])
        dummy_df.rename(columns={'1':'presence'})
    return(dummy_df)
#---



def configure_model( model, config,
                     window_type = None,
                     window_length_sec =None,
                     window_samples = None, 
                     overlap_samples = None,
                     overlap_fraction = None,
                     fft_size = None,

                     sample_rate = 16000,
                     random_trim = True,
                     bandpass_bypass = True,
                     time_mask_bypass = False,
                     frequency_mask_bypass = False,
                     add_noise_bypass = False,
                     rescale_bypass = False, 
                     random_affine_bypass = False,

                     lr = 0.01,
                     lr_cooling_factor = 0.7,
                     lr_update_interval = 10, #in epochs
                     momentum = 0.9, #leaky
                     weight_decay = 0.0005, #l2 regularization

                     height =224, width =224,
                     ):
    
#---
# Function to configure the model with hyperparameters specified in the configuration dict (below)
# Hyperparameter configuration consists of both pre-processing and model-specific hyperparameters
#---


    #Initiate preprocessor
    preprocessor = SpectrogramPreprocessor(sample_duration=3, height=height, width=width)
    
    #Spectrogram properties
    preprocessor.pipeline.to_spec.params.window_type = config['window_type'] or window_type
    preprocessor.pipeline.to_spec.params.window_samples = config['window_samples'] or window_samples
    preprocessor.pipeline.to_spec.params.window_length_sec  = config['window_length_sec'] or window_length_sec
    preprocessor.pipeline.to_spec.params.overlap_samples = config['overlap_samples'] or overlap_samples
    preprocessor.pipeline.to_spec.params.overlap_fraction = config['overlap_fraction'] or overlap_fraction
    preprocessor.pipeline.to_spec.params.fft_size = config['fft_size'] or fft_size

    #Audi0 loading properties
    preprocessor.pipeline.load_audio.params.sample_rate = config['sample_rate'] or sample_rate
    preprocessor.pipeline.random_trim_audio.params.random_trim = config['random_trim'] or random_trim
    preprocessor.pipeline.bandpass.bypass= config['bandpass_bypass'] or bandpass_bypass# Doesn't work with <22k hz samples
    preprocessor.pipeline.time_mask.bypass = config['time_mask_bypass'] or time_mask_bypass
    preprocessor.pipeline.frequency_mask.bypass = config['frequency_mask_bypass'] or frequency_mask_bypass
    preprocessor.pipeline.add_noise.bypass = config['add_noise_bypass'] or add_noise_bypass
    preprocessor.pipeline.rescale.bypass = config['rescale_bypass'] or rescale_bypass
    preprocessor.pipeline.random_affine.bypass = config['random_affine_bypass'] or random_affine_bypass

    model.preprocessor = preprocessor

    #model learning properties
    model.optimizer_params['lr']= config['lr'] or lr
    model.lr_cooling_factor = config['lr_cooling_factor'] or lr_cooling_factor
    model.lr_update_interval = config['lr_update_interval'] or lr_update_interval
    model.optimizer_params['momentum']= config['momentum'] or momentum
    model.optimizer_params['weight_decay']= config['weight_decay'] or weight_decay
    return(model)


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]

from torchvision import models

def get_model(config, device):

    # No weights - random initialization
    model = getattr(models, config.architecture)(weights=None)

    # Replace the last layer (number of classes; sigmoid)
    num_features = model.fc.in_features
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #model.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    model.fc = nn.Linear(num_features, out_features =1) #No need for sigmoid - located in loss func where it gives 'more numerically stable' results

    if config['dropout']:
        # Replace the fully connected layer with a new one that includes dropout
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_features, out_features =1)
            )


    model.to(device)

    return(model)



def get_preprocessor(config):
    preprocessor = SpectrogramPreprocessor(sample_duration=3, height=224, width=224)

    #Spectrogram properties
    preprocessor.pipeline.to_spec.params.window_type = config['window_type'] 
    preprocessor.pipeline.to_spec.params.window_samples = config['window_samples'] 
    preprocessor.pipeline.to_spec.params.window_length_sec  = config['window_length_sec'] 
    preprocessor.pipeline.to_spec.params.overlap_samples = config['overlap_samples'] 
    preprocessor.pipeline.to_spec.params.overlap_fraction = config['overlap_fraction']
    preprocessor.pipeline.to_spec.params.fft_size = config['fft_size'] #Audi0 loading properties
    preprocessor.pipeline.load_audio.params.sample_rate = config['sample_rate'] 
    preprocessor.pipeline.random_trim_audio.params.random_trim = config['random_trim']
    preprocessor.pipeline.bandpass.bypass= True  #Doesn't work with <22k hz samples
    preprocessor.pipeline.time_mask.bypass = config['time_mask_bypass']
    preprocessor.pipeline.frequency_mask.bypass = config['frequency_mask_bypass'] 
    preprocessor.pipeline.add_noise.bypass = config['add_noise_bypass'] 
    preprocessor.pipeline.rescale.bypass = config['rescale_bypass']
    preprocessor.pipeline.random_affine.bypass = config['random_affine_bypass']
    
    return(preprocessor)

def get_dataLoader(df, config, train=True):
    preprocessor = get_preprocessor(config)
    if train == True:
        dataset = AudioFileDataset(df,preprocessor)
    if train == False:
        dataset = AudioFileDataset(df,preprocessor)
        dataset.bypass_augmentations = True # Remove augmentations
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config['batch_size'], 
                                         shuffle=True,
                                         collate_fn = collate_audio_samples_to_tensors)    
    return dataloader




def get_TVdata(data_dir, balance:bool=False, n_balance = int):

#---
# Function to create metadata CSV for OpenSoundScape module
# Metadata csv consists of pandas dataframe:
    #Index of dataframe must be set as filepath to audiosamples
    #Columns of dataframe must be one-hot-encoded labels of species and/or positive/negative samples
#---

    #Define dict of filepaths and presence status (pos/neg) to be populated within loops
    filepath_presence_dict = {"filepath":[], "presence":[]}

    #Define path to main training data folder
    #weto_train_dir = os.path.join(data_dir, 'weto', 'train')
    weto_train_dir = os.path.join(data_dir, 'weto', 'train')

    #Iteratively loop through audio files in positive folder, then negative
    for set_key in ['positive', 'negative']:
        set_dir = os.path.join(weto_train_dir, set_key)

        #Create list of audio files in specified directory (pos/neg)
        filenames = os.listdir(set_dir)

        #Append filepaths and presence status to dict (empty, if first run through parent loop)
        filepath_list = filepath_presence_dict["filepath"]
        presence_list = filepath_presence_dict["presence"]

        #Loop through filenames to create list of filepaths and presence status
        for name in filenames:
            #Define filepath and append to list
            filepath = os.path.join(set_dir, name)
            filepath_list.append(filepath)

        #Append list to dict, as a value to 'filepath' key
        filepath_presence_dict.update({'filepath': filepath_list})

        #Create sequence of presence status values (pos/neg) of equal length to filepath list and append to dict, as a value to 'presence' key
        if set_key == "positive":
            presence_list = list(np.repeat('positive', len(filenames)))
            filepath_presence_dict.update({'presence': presence_list})
        if set_key == "negative":
            presence_list.extend(list(np.repeat('negative', len(filenames))))
            filepath_presence_dict.update({'presence': presence_list})       

    #Convert dict to dataframe and set index as filepath
    meta_weto = pd.DataFrame(filepath_presence_dict).set_index('filepath')

    #Use get_dummies to one-hot-encode the presence column
    meta_weto = pd.get_dummies(meta_weto, prefix="", prefix_sep='', columns=['presence'], dtype = int)

    #Define a list variable of the different presence statuses
    classes = meta_weto.columns

    #Split the data into a training (85%) and validation (15%) set
    train_df, valid_df = sklearn.model_selection.train_test_split(meta_weto, test_size=0.15, random_state=0)
    print(f"created train_df (len {len(train_df)}) and valid_df (len {len(valid_df)})")
 
    #Drop/repeat samples to the training set, if desired. THis is helpful to ensure balance among the classes (improves model performance)
    if balance:
        train_df = opensoundscape.data_selection.resample(train_df,n_samples_per_class=n_balance,random_state=0)
        print(f"There are {np.sum(train_df['positive']==1)} positive samples in train_df, after balancing")
        print(f"There are {np.sum(train_df['negative']==1)} negative samples in train_df, after balancing")
    else:
        print(f"created train_df (len {len(train_df)}) and valid_df (len {len(valid_df)})")
        print(f"There are {np.sum(train_df['positive']==1)} positive samples in train_df")
        print(f"There are {np.sum(train_df['negative']==1)} negative samples in train_df")
        print(f"The dataset consists of {len(classes)} class(es)")


    return(train_df, valid_df)
#---


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_acc = None
        self.best_prc = None
        self.best_rec = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, val_acc, val_prc, val_rec):
        score = -val_loss
        acc = val_acc
        prc = val_prc
        rec = val_rec
        if self.best_score is None:
            self.best_score = score
            self.best_acc = acc
            self.best_prc = prc
            self.best_rec = rec
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("Updating best model... \n")
            self.best_score = score
            self.best_acc = acc
            self.best_prc = prc
            self.best_rec = rec
            self.counter = 0


class ConfigureResnet(nn.Module):
    def __init__(self, architecture, dropout:bool, dropout_rate=0.5):
        super(ConfigureResnet, self).__init__()
        # Load a pretrained ResNet model
        self.resnet = getattr(models, architecture)(weights=None)
        
        num_features = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if dropout:
        # Replace the fully connected layer with a new one that includes dropout
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, out_features =1)
            )
        else:
            self.resnet.fc = nn.Linear(num_features, out_features =1) 

    def forward(self, x):
        return self.resnet(x)