import torch


import random

import os

import numpy as np
import pickle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras import models, layers, optimizers, callbacks

from sample_enumerate_abstraction_pedagogical_ing_sonar_dataset import *

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

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        num_input_features=60

       
       
       
        num_output_features=1

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(num_input_features,120),
            torch.nn.ReLU(),
            torch.nn.Linear(120,120),
            torch.nn.ReLU(),
            torch.nn.Linear(120,30),
            torch.nn.ReLU(),
            torch.nn.Linear(30,num_output_features),
            torch.nn.Sigmoid()

            )

    def forward(self,x):
        x = self.flatten(x)
        y=self.linear_relu_stack(x)

        return y


if __name__ == "__main__":

    model=Model()

    model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))
        
    data = pd.read_csv(other_data_name+'/sonar.all-data.csv', header=None)



    data= data.replace(to_replace="R", value=0, regex=True)
    data= data.replace(to_replace="M", value=1, regex=True)

    # print(data.head())
    # sys.exit()
    X = np.array(data.iloc[:,:-1]).astype('float32')

    # print(X.shape)

    y = np.array(data.iloc[:,-1]).astype('float32')

    shuffle=np.random.permutation(X.shape[0])
    X=X[shuffle,:]
    y=y[shuffle]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # np.save(other_data_name+'/sonar_x_train.npy', X_train)
    # np.save(other_data_name+'/sonar_y_train.npy', Y_train)
    # np.save(other_data_name+'/sonar_x_test.npy', X_test)
    # np.save(other_data_name+'/sonar_y_test.npy', Y_test)
    
    # X_train = np.load(other_data_name+'/sonar_x_train.npy',allow_pickle=True)
    # Y_train = np.load(other_data_name+'/sonar_y_train.npy',allow_pickle=True)
    
    # X_test  = np.load(other_data_name+'/sonar_x_test.npy',allow_pickle=True)
    
    # Y_test  = np.load(other_data_name+'/sonar_y_test.npy',allow_pickle=True)

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

    np.save(other_data_name+'/logical_rules_sonar_data_2.npy',ruleSet)
    # print(f'ruleSet shape: {ruleSet.shape}')


    
        
        
        
        
        

        
        
        
        
        
        
        
        
        
