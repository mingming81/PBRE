import torch


import random

import os

import numpy as np
import pickle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from tensorflow.keras import models, layers, optimizers, callbacks

from sample_enumerate_abstraction_pedagogical_ing_bcw_dataset_2 import *

from joblib import dump, load

import pickle

random.seed(0)

torch.autograd.set_detect_anomaly(True)

# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

from joblib import dump, load

# %%
# create a data storing path

dirName='./../v3/data2'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    

other_data_name='./../v3/data2'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)

print('beginnnnnnnnnnnnnnnnn')

class ServiceModel(torch.nn.Module):
    def __init__(self):
        super(ServiceModel,self).__init__()
        
        num_input_features=30

       
        num_hidden=30 
       
        num_output_features=1

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(num_input_features,num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden,30),
            torch.nn.ReLU(),
            torch.nn.Linear(30,30),
            torch.nn.ReLU(),
            torch.nn.Linear(30,num_output_features),
            torch.nn.Sigmoid()

            )

      
    def forward(self,x):
        x = self.flatten(x)
        y=self.linear_relu_stack(x)

        return y


if __name__ == "__main__":

    model=ServiceModel()

    model.load_state_dict(torch.load(other_data_name+'/bcw_nn_classifier.pth'))
        
    sc=load(other_data_name+'/bcw_std_scaler.bin')
    
    X_train = np.load(other_data_name+'/bcw_x_train.npy',allow_pickle=True)
    Y_train = np.load(other_data_name+'/bcw_y_train.npy',allow_pickle=True)
    
    X_test  = np.load(other_data_name+'/bcw_x_test.npy',allow_pickle=True)
    
    Y_test  = np.load(other_data_name+'/bcw_y_test.npy',allow_pickle=True)

    # X_train=torch.from_numpy(X_train).type(torch.float32)
    # Y_train=torch.from_numpy(Y_train).type(torch.float32)

# %%
            
    seapeda=Seapeda()
     
    numRefineTree=0
    
    lenX=X_train.shape[0]
    
    print(f'lenX: {lenX}')
    
    for i in range(lenX):
        
        print(f'state idx: {i}')
   
        x=X_train[[i]]
        
        rule=seapeda.ruleExtract(x,model)
      
        seapeda.update(rule)

        seapeda.pruneNodes()
        
    
    rule=seapeda.refineRule()
    print('refined rule:\n ',rule.shape)
    print("\n===========================\n")

    ruleSet=copy.deepcopy(rule)
    
       

    # %%%%%%%%%%%%%%%%%%%%%%%%%
    # 10 epochs

    np.save(other_data_name+'/logical_rules_bcw_data_2.npy',ruleSet)
    # print(f'ruleSet shape: {ruleSet.shape}')


    
        
        
        
        
        

        
        
        
        
        
        
        
        
        
