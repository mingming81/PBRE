

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
import random

import os

import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sample_enumerate_abstraction_pedagogical_ing_iris_dataset_4 import *

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

        self.layer1=nn.Linear(input_dim,30)
        self.layer2=nn.Linear(30,30)
        self.layer3=nn.Linear(30,30)
        self.layer4=nn.Linear(30,3)

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

  
    
    model=Model(4)
    model.load_state_dict(torch.load(other_data_name+'/iris_nn_classifier.pth'))

    data = pd.read_csv(other_data_name+'/Iris.csv')


    X = data.drop(['Id', 'Species'], axis=1).values

    d={'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

    data['Species']=data['Species'].map(d)

    y = data['Species'].values


    # X = sc.fit_transform(XX)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)

    # np.save(other_data_name+'/iris_trainsetX_5.npy', X_train)
    # np.save(other_data_name+'/iris_trainsetY_5.npy', Y_train)
    # np.save(other_data_name+'/iris_testsetX_5.npy', X_test)
    # np.save(other_data_name+'/iris_testsetY_5.npy', Y_test)
    # np.save(other_data_name+'/iris_valsetX_5', X_val)
    # np.save(other_data_name+'/iris_valsetY_5.npy', Y_val)

    X_train=np.load(other_data_name+'/iris_trainsetX_5.npy')
    Y_train=np.load(other_data_name+'/iris_trainsetY_5.npy')
    X_test=np.load(other_data_name+'/iris_testsetX_5.npy')
    Y_test=np.load(other_data_name+'/iris_testsetY_5.npy')
    X_val=np.load(other_data_name+'/iris_valsetX_5.npy')
    Y_val=np.load(other_data_name+'/iris_valsetY_5.npy')

    # X_train=torch.from_numpy(X_train).type(torch.float32)
    # Y_train=torch.from_numpy(Y_train)    

    # sys.exit()
        

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
        
      

    np.save(other_data_name+'/logical_rules_iris_data_5.npy',ruleSet)
    print(f'ruleSet shape: {ruleSet.shape}')


    
        
        
        
        
        

        
        
        
        
        
        
        
        
        
