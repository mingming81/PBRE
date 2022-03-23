

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
import random

import os

import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sample_enumerate_abstraction_pedagogical_ing_heart_dataset_2 import *

import pandas as pd

random.seed(0)

torch.autograd.set_detect_anomaly(True)

# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

# %%
# create a data storing path


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()

        self.layer1=nn.Linear(input_dim,50)
        self.layer2=nn.Linear(50,50)
        self.layer3=nn.Linear(50,20)
        self.layer4=nn.Linear(20,5)

    def forward(self,x):
        # x = torch.nn.Flatten(x)
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x = F.softmax(self.layer4(x), dim=1)

        return x






dirName='./../v3/data2'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='./../v3/data2'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)

print('beginnnnnnnnnnnnnnnnn')



if __name__ == "__main__":

  
    
    model=Model(13)
    model.load_state_dict(torch.load(other_data_name+'/heart_nn_classifier.pth'))

    data = pd.read_csv(other_data_name+'/heart_clearned_df.csv')


    X = np.load(other_data_name+'/heart_x.npy', allow_pickle=True)
    y = np.load(other_data_name+'/heart_y.npy', allow_pickle=True)


    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train=np.load(other_data_name+'/heart_train_x.npy',allow_pickle=True)
    Y_train=np.load(other_data_name+'/heart_train_y.npy',allow_pickle=True)
    X_test=np.load(other_data_name+'/heart_test_x.npy',allow_pickle=True)
    Y_test=np.load(other_data_name+'/heart_test_y.npy',allow_pickle=True)

    # np.save(other_data_name+'/heart_train_x.npy', X_train)
    # np.save(other_data_name+'/heart_train_y.npy', Y_train)
    # np.save(other_data_name+'/heart_test_x.npy', X_test)
    # np.save(other_data_name+'/heart_test_y.npy', Y_test)

    # X_train=torch.from_numpy(X_train).type(torch.float32)
    # Y_train=torch.from_numpy(Y_train)    
        

    seapeda=Seapeda()
    # %%
    
    numRefineTree=0

    ruleSet=np.full([1,5],None)

    lenX=X_train.shape[0]

    print(f'lenX: {lenX}')

    # import sys

    # sys.exit()
    
    for i in range(lenX):
        
        print(f'state idx: {i}')
   
        x=X_train[[i],:]
        
        # x=x.to(device)
        
        
        rule=seapeda.ruleExtract(x,model)
        # print('extract rule:\n ',rule)
        # print("\n===========================\n")
        
        seapeda.update(rule)
        
     

        
    rule=seapeda.refineRule()
        # print('refined rule:\n ',rule)
        # print("\n===========================\n")
        
        
        
        # if ruleSet[0,0]==None:
    ruleSet=copy.deepcopy(rule)
        
      

    np.save(other_data_name+'/logical_rules_heart_data_1000_1000_2.npy',ruleSet)
    print(f'ruleSet shape: {ruleSet.shape}')


    
        
        
        
        
        

        
        
        
        
        
        
        
        
        
