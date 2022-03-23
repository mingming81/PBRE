# it in fact extractes rules from the training data, the function of the neural network is to remove
# insignificant attributes
# OK
# %%
import torch


import random

import os

import numpy as np
import pickle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from tensorflow.keras import models, layers, optimizers, callbacks

import torch.nn.functional as F
import torch.nn as nn
# from sample_enumerate_abstraction_pedagogical_ing_australian_dataset import *

from joblib import dump, load

import pickle

import copy

import sys

random.seed(0)


torch.autograd.set_detect_anomaly(True)

# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

from joblib import dump, load

from sklearn.model_selection import train_test_split

# %%
# create a data storing path

dirName='./v3/data2'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    

other_data_name='./v3/data2'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)
    
# %%
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


model=Model()

model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))

# %%

X_train=np.load(other_data_name+'/sonar_x_train_5.npy')
Y_train=np.load(other_data_name+'/sonar_y_train_5.npy')
X_test=np.load(other_data_name+'/sonar_x_test_5.npy')
Y_test=np.load(other_data_name+'/sonar_y_test_5.npy')
X_val=np.load(other_data_name+'/sonar_x_val_5.npy')
Y_val=np.load(other_data_name+'/sonar_y_val_5.npy')


logicalRules=np.load(other_data_name+'/logical_rules_sonar_data.npy',allow_pickle=True)
print(f"extracted rule shape: {logicalRules.shape}")
    
X_train=torch.from_numpy(X_train).type(torch.float32)

# %%

T_x=np.full([1,X_train.shape[1]],None)
# Tidx=np.full([1,1],None)
T_y=np.full([1],None)

for i in range(X_train.shape[0]):
    
    x=copy.deepcopy(X_train[[i]])
    
    y_pred=model(x).reshape((1,-1))
    y_pred=y_pred.reshape(-1).detach().numpy().astype(np.float32).round()
    
    
    y_true=Y_train[[i]].astype(np.float32)
    
    if y_pred==y_true:
        
        if T_x[0,0]==None:
            T_x=copy.deepcopy(np.array(X_train[[i]]).astype('float32'))
        else:
            T_x=np.concatenate((T_x,copy.deepcopy(np.array(X_train[[i]]).astype('float32'))),axis=0)
            
        if T_y[0]==None:
            T_y=copy.deepcopy(y_true)
        else:
            T_y=np.concatenate((T_y,copy.deepcopy(y_true)))
            
# %%
# pruning
# E_x=[]
# E_y=[]
# err=[]
# for i in range(T_x.shape[1]):
    
#     print('i: ',i)
    
#     Ei_x=np.full([1,X_train.shape[1]],None)
#     Ei_y=np.full([1],None)
    
#     model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))
    
#     for j in range(T_x.shape[0]):
#         # print('j: ',j)
        
#         x=copy.deepcopy(T_x[[j]])
        
#         x=torch.from_numpy(x).type(torch.float32)
        
#         keyv=list(model.state_dict().keys())
#         model.state_dict()[keyv[0]][:,[i]]=torch.zeros(model.state_dict()[keyv[0]][:,[i]].shape)
        
#         y_pred=model(x).reshape((1,-1))
#         y_pred=y_pred.reshape(-1).detach().numpy().astype(np.float32).round()
        
        
#         y_true=T_y[[j]].astype(np.float32)
        
#         if y_pred!=y_true:
            
#             if Ei_x[0,0]==None:
#                 Ei_x=copy.deepcopy(np.array(T_x[[j]]).astype("float32"))
#             else:
#                 Ei_x=np.concatenate((Ei_x,copy.deepcopy(np.array(T_x[[j]]).astype("float32"))),axis=0)
            
#             if Ei_y[0]==None:
#                 Ei_y=copy.deepcopy(y_true)
#             else:
#                 Ei_y=np.concatenate((Ei_y,copy.deepcopy(y_true)))
                
                
    
# #     E_x.append(copy.deepcopy(Ei_x))
    
        
    
# #     E_y.append(copy.deepcopy(Ei_y))
    
# #     err.append(Ei_x.shape[0])

# # %%
# for i in range(len(err)):
    
#     print(f'i: {i}')
    
#     print(err[i])  #--> no attributes are deleted

# with open(other_data_name+"/E_x.txt", "wb") as fp:   #Pickling
#     pickle.dump(E_x, fp)
   
# with open(other_data_name+"/E_y.txt", "wb") as fp:   #Pickling
#     pickle.dump(E_y, fp)
   
# with open(other_data_name+"/E_x.txt", "wb") as fp:   #Pickling
#     pickle.dump(err, fp)

# %%
# pruning
# with open(other_data_name+"/E_x.txt", "wb") as fp:   #Pickling
#    E_x=pickle.load(fp)
   
# with open(other_data_name+"/E_y.txt", "wb") as fp:   #Pickling
#    E_y=pickle.load(fp)
   
# with open(other_data_name+"/E_x.txt", "wb") as fp:   #Pickling
#    err=pickle.load(fp)
   
# %

            
# %%
# pruning

# testing on test dataset
model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))
x_test=torch.from_numpy(np.array(X_test)).type(torch.float32)
predicted=model(x_test)

predicted=predicted.reshape(-1).detach().numpy().astype(np.float32).round()

Y_test=Y_test.astype(np.float32)
acc=(predicted==Y_test).mean()

print(acc)

# %%
# pruning
# input node 0 is removed
keyv=list(model.state_dict().keys())

# for k in range(60):
    
#     model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))
#     model.state_dict()[keyv[0]][:,[k]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
#     x_test=torch.from_numpy(np.array(X_test)).type(torch.float32)
#     predicted=model(x_test)

#     predicted=predicted.reshape(-1).detach().numpy().astype(np.float32).round()

#     Y_test=Y_test.astype(np.float32)
#     acc=(predicted==Y_test).mean()
#     print(f'k: {k}, acc: {acc}')
   

model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))
model.state_dict()[keyv[0]][:,[0]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[1]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[2]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[3]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[4]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[5]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[6]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[7]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[8]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[9]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[13]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[14]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[32]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[33]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[34]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[36]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[37]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[38]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[39]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[41]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[42]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[46]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[49]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[50]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[51]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[52]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[53]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[54]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[55]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[56]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[57]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[58]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
model.state_dict()[keyv[0]][:,[59]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[50]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[51]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[52]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[53]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[54]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[55]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[56]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[57]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[58]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)
# model.state_dict()[keyv[0]][:,[59]]=torch.zeros(model.state_dict()[keyv[0]][:,[0]].shape)


x_test=torch.from_numpy(np.array(X_test)).type(torch.float32)
predicted=model(x_test)

predicted=predicted.reshape(-1).detach().numpy().astype(np.float32).round()

Y_test=Y_test.astype(np.float32)
acc=(predicted==Y_test).mean()

print(acc)

# model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))

# %%

notsignificant=np.array([0,1,2,3,4,5,6,7,8,9,13,14,32,33,34,36,37,38,39,41,42,46,49,50,51,52,53,54,55,56,57,58,59])

# %%

# Data range computation
Range=np.full([X_train.shape[1], len(set(Y_train))],None)

classes=list(set(Y_train))

classes=sorted(classes, key=lambda x: x)

for c in range(len(classes)):
    
    C_x=np.full([1,T_x.shape[1]],None)
    
    for i in range(T_x.shape[0]):
        
        if T_y[i]==classes[c]:
            
            if C_x[0,0]==None:
                C_x=copy.deepcopy(np.array(T_x[[i]]).astype('float32'))
            
            else:
                C_x=np.concatenate((C_x,copy.deepcopy(np.array(T_x[[i]]).astype('float32'))),axis=0)
                
    for j in range(T_x.shape[1]):
        
        rg=np.zeros((1,2)).astype('float32')
        
        C_xj=np.array(sorted(C_x, key=lambda x: x[j])).astype('float32')
        
        rg[0,0]=C_xj[0,j]
        rg[0,1]=C_xj[-1,j]
        
        Range[j,c]=rg
        

# %%

# rule pruning 1: remove only one condition each time
# random.seed(1)
num_corr=0

x_test=copy.deepcopy(X_test)
y_test=copy.deepcopy(Y_test)

# model.load_state_dict(torch.load(other_data_name+'/bcw_nn_classifier.pth'))

xx_test=torch.from_numpy(np.array(x_test)).type(torch.float32)
predicted=model(xx_test)

predicted=predicted.reshape(-1).detach().numpy().astype(np.float32).round()


yy_test=copy.deepcopy(y_test.astype(np.float32))
acc=(predicted==yy_test).mean()
print(acc)


RangeM=np.full([61,2],None)
RangeM[:60,:]=copy.deepcopy(Range)
RangeM[-1,0]=np.array([[0,0]]).astype('float32')
RangeM[-1,1]=np.array([[1,1]]).astype('float32')


np.random.seed(0)
shuffle=np.random.permutation(RangeM.shape[1])
RangeD=copy.deepcopy(RangeM[:,shuffle])


# RangeD=copy.deepcopy(RangeM)


num_corr=0

num_none=0

num_sim=0

for i in range(x_test.shape[0]):
    
    # print(f'i: {i}')
    
    y_rule=None
    y_true=y_test[[i]].astype('float32')
    
    y_pred=predicted[[i]].astype('float32')
    
    cpr=0
    
    for j in range(2):
        # print(f'j: {j}')
        
        num_p=0
        
        lenlennotIdx=0
        
        for k in range(x_test.shape[1]):
            
            # print(T_x[i,k], Range[k,j][0,0], Range[k,j][0,1])
            exitR=False
            
            if k in notsignificant:
                exitR=True
                
            if exitR:
                continue
            
        
            if x_test[i,k]>=RangeD[k,j][0,0] and x_test[i,k]<=RangeD[k,j][0,1]:
                
                num_p+=1
            else:
                break
                
        
        if notsignificant[0]==None:
            lenlennotIdx=0
        else:
            lenlennotIdx=notsignificant.shape[0]
        
        
        if num_p==(x_test.shape[1]-lenlennotIdx) and (num_p!=0):
            
            y_rule=RangeD[-1,j][0,0]
            
            # print(f'same: {j}')
            
            cpr+=1
   
            if y_true==y_rule:
                
                num_corr+=1
            
            if y_rule==y_pred:
                num_sim+=1
                    
            break
        
    if y_rule==None:
        num_none+=1
        
    if cpr>1:
        print(f'cpr: {cpr}')
        print('mul i: ', i)
        # sys.exit()
        
        

accTe=num_corr/x_test.shape[0] 
accNone=num_none/x_test.shape[0]
accSim=num_sim/x_test.shape[0]
       
print(f'accuracy: {num_corr}/{accTe}')
print(f'accNone: {num_none} / {accNone}')
print(f'accSim: {num_sim} / {accSim}')

# %%
# rule pruning 2

RangeM=np.full([61,2],None)
RangeM[:60,:]=copy.deepcopy(Range)
RangeM[-1,0]=np.array([[0,0]]).astype('float32')
RangeM[-1,1]=np.array([[1,1]]).astype('float32')


np.random.seed(0)
shuffle=np.random.permutation(RangeM.shape[1])
RangeD=copy.deepcopy(RangeM[:,shuffle])


# RangeD=copy.deepcopy(RangeM)

x_test=copy.deepcopy(X_test)
y_test=copy.deepcopy(Y_test)


model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))

xx_test=torch.from_numpy(np.array(x_test)).type(torch.float32)
predicted=model(xx_test)

predicted=predicted.reshape(-1).detach().numpy().astype(np.float32).round()



def rulePrune(notIdx, Rangec):
  
    num_corr=0

    num_none=0

    num_sim=0
    
    repeat=0
    
    for i in range(x_test.shape[0]):
        
        # print(f'=========')
        
        # print(f'i: {i}')
        
        y_rule=None
        y_true=y_test[[i]].astype('float32')
        y_pred=predicted[[i]].astype('float32')
        
        cpr=0
        
        for j in range(2):
            
            # print(f'j: {j}')
            
            num_p=0
            
            for k in range(x_test.shape[1]):
                
                exitR=False
                
                if k in notsignificant:
                    exitR=True
                
                
                
                if (k in notIdx[0,Rangec[-1,j].astype('int')[0,0]]):
                        exitR=True

                
                if exitR:
                    continue
                
                if x_test[i,k]>=Rangec[k,j][0,0] and x_test[i,k]<=Rangec[k,j][0,1]:
                    
                    num_p+=1 
                else: 
                    break
                
            if notsignificant[0]==None:
                lennotsignificant=0
            else:
                lennotsignificant=notsignificant.shape[0]
                
            if notIdx[0,Rangec[-1,j].astype('int')[0,0]][0,0]==None:
                lennotIdx=0
            else:
                lennotIdx=notIdx[0,Rangec[-1,j].astype('int')[0,0]].shape[1]
                
                
            
            if (num_p==(x_test.shape[1]-lennotIdx-lennotsignificant)) and (num_p!=0):
                # print(f'same: {j}')
                cpr+=1
                y_rule=Rangec[-1,j][0,0]
            
                if y_true==y_rule:
                    num_corr+=1
                    
                if y_rule==y_pred:
                    num_sim+=1

                break
            
        if y_rule==None:
            num_none+=1
        
        if cpr>1:
            # print(f'mul: {i}')
            repeat+=1
            
            
        # print('fired rules number: ',cpr)
                
    accTe=num_corr/x_test.shape[0] 
    accNone=num_none/x_test.shape[0]
    accSim=num_sim/x_test.shape[0]
    
    # print(f'accuracy: {num_corr}/{accTe}')
    # print(f'accNone: {num_none} / {accNone}')
    # print(f'accSim: {num_sim} / {accSim}')
    
    return accTe



# notIdx=np.array([2,2],None)
# notIdx[0,0]=0
# notIdx[0,1]=13
# notIdx[1,0]=1
# notIdx[1,1]=13

# test for each rule, each time remove one condition from this ORIGINAL rule.



print(f'k: {k}')

notIdx=np.full([1,2],None)

notIdx[0,0]=np.array([[10,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,35,40,43,44,45,47]])

notIdx[0,1]=np.array([[10,11,12,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,35,40,43,44,45,47]])
    

accTremove=rulePrune(notIdx,RangeD)

print(f'accTremove: {accTremove}')




        
# %%

# rule update

num_corr=0

x_test=copy.deepcopy(X_test)
y_test=copy.deepcopy(Y_test)

classes=list(set(Y_train))

classes=sorted(classes, key=lambda x: x)

        

select_idxT=np.full([999,999],None)

for i in range(T_x.shape[0]):
    
    y_rule=None
    y_true=T_y[[i]].astype('float32')
    
    num_r=0
    
    select_idx=np.full([1,2],None)
    select_r=np.full([1,1],None)
    
    for j in range(Range.shape[1]):
        
        num_p=0
        
        for k in range(T_x.shape[1]):
            
            exitR=False
            
            if k in notsignificant:
                exitR=True
            
            
            
            if (k in notIdx[0,RangeD[-1,j].astype('int')[0,0]]):
                    exitR=True
            
            if exitR:
                continue
        
            if T_x[i,k]>=Range[k,j][0,0] and T_x[i,k]<=Range[k,j][0,1]:
                
                num_p+=1
                
            else:
                break
            
        if notsignificant[0]==None:
            lennotsignificant=0
        else:
            lennotsignificant=notsignificant.shape[0]
            
        if notIdx[0,RangeD[-1,j].astype('int')[0,0]][0,0]==None:
            lennotIdx=0
        else:
            lennotIdx=notIdx[0,RangeD[-1,j].astype('int')[0,0]].shape[1]
    
        if (num_p==(T_x.shape[1]-lennotIdx-lennotsignificant)) and (num_p!=0):
            
            # print(f'i: {i}')
            
            num_r+=1
    
            if select_r[0,0]==None:
                select_r=copy.deepcopy(np.array([[j]]).astype('float32'))
                
            else:
                select_r=np.concatenate((select_r,copy.deepcopy(np.array([[j]]).astype('float32'))),axis=1)
    
    if num_r>1:
             
        select_idx[0,0]=copy.deepcopy(select_r)
        select_idx[0,1]=np.array([[i]]).astype('float32')
        
        if select_idxT.shape[0]==999:
            select_idxT=copy.deepcopy(select_idx)
        else:
            select_idxT=np.concatenate((select_idxT,copy.deepcopy(select_idx)),axis=0)
                  

# rule update
passedIdx=np.full([1,1],None)

for i in range(select_idxT.shape[0]-1):
    
    if passedIdx[0,0]!=None:
        if i in passedIdx:
            continue
        
    for j in range(i+1, select_idxT.shape[0]):
        
        if (select_idxT[i,0][0,0] in select_idxT[j,0]) and (select_idxT[i,0][0,1] in select_idxT[j,0]):
            
            select_idxT[i,1]=np.concatenate((select_idxT[i,1], select_idxT[j,1]),axis=1)
            
            if passedIdx[0,0]==None:
                passedIdx=np.array([[j]]).astype('float32')
                
            else:
                passedIdx=np.concatenate((passedIdx,np.array([[j]]).astype('float32')),axis=1)
                

select_idxT=np.delete(select_idxT, passedIdx.astype('int'), axis=0)
                  

# %%
# rule update

def refineRule(AttrList, Rangec):
    
    
  

    for i in range(select_idxT.shape[0]):
        
        class_count=np.zeros([1,select_idxT[0,1].shape[1]])
        
        
        for j in range(select_idxT[i,1].shape[1]):
            
            if T_y[select_idxT[i,1][0].astype('int')[j]]==0:
                class_count[0,0]+=1
            else:
                class_count[0,1]+=1
           
        T_x_int=T_x[select_idxT[i,1][0].astype('int')]
        
        if AttrList[0]!=None:
            
            ruleIntk=Rangec[select_idxT[i,0][0].astype('int'),:]
        
            for k in AttrList:
                
                exitR=False
                
                if k in notsignificant:
                    exitR=True
                      
                # if (k in notIdx[0,Rangec[j,-1].astype('int')[0,0]]):
                #         exitR=True
                
                if exitR:
                    continue
                
                T_x_int=np.array(sorted(T_x_int, key=lambda x: x[k])).reshape(T_x_int.shape)
                
                
                
                ruleIntk=np.array(sorted(ruleIntk, key=lambda x: x[k][0,0])).reshape(ruleIntk.shape)
            
                if class_count[0,0]>class_count[0,1]:
             
                    if ruleIntk[1,k][0,0]<ruleIntk[0,k][0,1]:
                        
                        if ruleIntk[1,k][0,1]>ruleIntk[0,k][0,1]:
                            
                            # print(f'k: {k}')
                        
                            if ruleIntk[0,-1][0,0]==0:
                                
                                ruleIntk[1,k][0,0]=T_x_int[-1,k]
                                # ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1],T_x_int[-1,k],ruleIntk[1,k][0,1])+epsilon
                                
                                # ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0], T_x_int[0,k])
                                # ruleIntk[0,k][0,1]=max(ruleIntk[0,k][0,1], T_x_int[-1,k])
                                
                            else:
                                ruleIntk[0,k][0,1]=T_x_int[0,k]
                                # ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0],T_x_int[0,k],ruleIntk[1,k][0,0])-epsilon
                                
                                # ruleIntk[1,k][0,0]=min(ruleIntk[1,k][0,0], T_x_int[0,k])
                                # ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1], T_x_int[-1,k])
                                
                            #     ruleIntk[1,k][0,0]=T_x_int[-1,k]+epsilon
                            #     ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1],T_x_int[-1,k],ruleIntk[1,k][0,1])+epsilon
                                
                            #     ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0], T_x_int[0,k])
                            #     ruleIntk[0,k][0,1]=max(ruleIntk[0,k][0,1], T_x_int[-1,k])
                                
                            # else:
                            #     ruleIntk[0,k][0,1]=T_x_int[0,k]-epsilon
                            #     ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0],T_x_int[0,k],ruleIntk[1,k][0,0])-epsilon
                                
                            #     ruleIntk[1,k][0,0]=min(ruleIntk[1,k][0,0], T_x_int[0,k])
                            #     ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1], T_x_int[-1,k])
                        
                        else:
                            
                            if ruleIntk[0,-1][0,0]==0:
                                ruleIntk=np.delete(ruleIntk, np.array([[1]]), axis=0)
                                
                            else:
                                newruleIntk=copy.deepcopy(ruleIntk[[0]])
                                
                                ruleIntk[0,k][0,1]=T_x_int[0,k]
                         
                                newruleIntk[0,k][0,0]=T_x_int[-1,k]
                                
                                ruleIntk=np.concatenate((ruleIntk, newruleIntk), axis=0)
                                
            
                elif class_count[0,0]<class_count[0,1]:
                    
                    if ruleIntk[1,k][0,0]<ruleIntk[0,k][0,1]:
                        
                        if ruleIntk[1,k][0,1]>ruleIntk[0,k][0,1]:
                        
                            if ruleIntk[0,-1][0,0]==0:
                                
                                ruleIntk[0,k][0,1]=T_x_int[0,k]
                                # ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0],T_x_int[0,k],ruleIntk[1,k][0,0])-epsilon
                                
                                # ruleIntk[1,k][0,0]=min(ruleIntk[1,k][0,0], T_x_int[0,k])
                                # ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1], T_x_int[-1,k])
                            
                            else:
                                
                                ruleIntk[1,k][0,0]=T_x_int[-1,k]
                                # ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1],T_x_int[-1,k],ruleIntk[1,k][0,1])+epsilon
                                
                                # ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0], T_x_int[0,k])
                                # ruleIntk[0,k][0,1]=max(ruleIntk[0,k][0,1], T_x_int[-1,k])
                                
                            #     ruleIntk[0,k][0,1]=T_x_int[0,k]-epsilon
                            #     ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0],T_x_int[0,k],ruleIntk[1,k][0,0])-epsilon
                                
                            #     ruleIntk[1,k][0,0]=min(ruleIntk[1,k][0,0], T_x_int[0,k])
                            #     ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1], T_x_int[-1,k])
                            
                            # else:
                                
                            #     ruleIntk[1,k][0,0]=T_x_int[-1,k]+epsilon
                            #     ruleIntk[1,k][0,1]=max(ruleIntk[1,k][0,1],T_x_int[-1,k],ruleIntk[1,k][0,1])+epsilon
                                
                            #     ruleIntk[0,k][0,0]=min(ruleIntk[0,k][0,0], T_x_int[0,k])
                            #     ruleIntk[0,k][0,1]=max(ruleIntk[0,k][0,1], T_x_int[-1,k])
                                
                        else:
                            if ruleIntk[0,-1][0,0]==0:
                                
                                newruleIntk=copy.deepcopy(ruleIntk[[0]])
                                
                                ruleIntk[0,k][0,1]=T_x_int[0,k]
                         
                                newruleIntk[0,k][0,0]=T_x_int[-1,k]
                                
                                ruleIntk=np.concatenate((ruleIntk, newruleIntk), axis=0)
                                
                            else:
                                
                                if ruleIntk[0,-1][0,0]==0:
                                    ruleIntk=np.delete(ruleIntk, np.array([[1]]), axis=0)
                                
                            
            Rangec=np.concatenate((Rangec, copy.deepcopy(ruleIntk)),axis=0)
            
        
            toDeleteIdx=np.full([1,1],None)
            
            for i in range(select_idxT.shape[0]):
                
                if toDeleteIdx[0,0]==None:
                    toDeleteIdx=copy.deepcopy(select_idxT[i,0]).astype('int')
                    
                else:
                    toDeleteIdx=np.contenate((toDeleteIdx,copy.deepcopy(select_idxT[i,0]).astype('int')),axis=1)
                    
            if toDeleteIdx[0,0]!=None:
                Rangec=np.delete(Rangec,toDeleteIdx, axis=0)
            
    return Rangec
                    
# %%
# rule update
# x_test=copy.deepcopy(X_test)
# y_test=copy.deepcopy(Y_test)

# x_test=copy.deepcopy(X_train)
# y_test=copy.deepcopy(Y_train)

x_test=copy.deepcopy(X_val)
y_test=copy.deepcopy(Y_val)

model.load_state_dict(torch.load(other_data_name+'/sonar_nn_classifier.pth'))

xx_test=torch.from_numpy(np.array(x_test)).type(torch.float32)
predicted=model(xx_test)

predicted=predicted.reshape(-1).detach().numpy().astype(np.float32).round()


def rulePrune2(notIdx, Rangec):
  
    num_corr=0
    
    num_none=0
    
    num_sim=0
    
    for i in range(x_test.shape[0]):
        
        y_rule=None
        y_true=y_test[[i]].astype('float32')
        
        y_pred=predicted[[i]].astype('float32')
        
        cpr=0
        
        for j in range(Rangec.shape[0]):
            
            num_p=0
            
            for k in range(x_test.shape[1]):
                
                exitR=False
                
                if k in notsignificant:
                    exitR=True
                
                
                
                if (k in notIdx[0,Rangec[j,-1].astype('int')[0,0]]):
                        exitR=True
                        
                        
                
                if (k in notIdx[0,Rangec[j,-1].astype('int')[0,0]]):
                    exitR=True
                    
                
                if exitR:
                    continue
                
                if x_test[i,k]>=Rangec[j,k][0,0] and x_test[i,k]<=Rangec[j,k][0,1]:
                    
                    num_p+=1 
                else: 
                    break
            
            if notIdx[0,Rangec[j,-1].astype('int')[0,0]][0,0]==None:
                lennotIdx=0
            else:
                lennotIdx=notIdx[0,Rangec[j,-1].astype('int')[0,0]].shape[1]
                
            if notsignificant[0]==None:
                lennotsignificant=0
            else:
                lennotsignificant=notsignificant.shape[0]
                
            
            if (num_p==(x_test.shape[1]-lennotIdx-lennotsignificant)) and (num_p!=0):
                cpr+=1
                y_rule=Rangec[j,-1][0,0]
            
                if y_true==y_rule:
                    num_corr+=1
                    
                if y_rule==y_pred:
                    num_sim+=1
                        
                break
            
        if y_rule==None:
            num_none+=1
 
            
    accTremove=num_corr/x_test.shape[0]
    accNone=num_none/x_test.shape[0]
    accSim=num_sim/x_test.shape[0]
    
    print(f'accuracy: {num_corr}/{accTremove}')
    print(f'accNone: {num_none} / {accNone}')
    print(f'accSim: {num_sim} / {accSim}')
    
    
    return accTremove


# %%
# %% =============1==================



accTec=copy.deepcopy(accTe)

toDecideList=np.array([i for i in range(x_test.shape[1])]).astype('int')

RangeC=copy.deepcopy(np.transpose(RangeD))

RangeK=copy.deepcopy(RangeC)

for k4 in range(1):
# for k4 in range(1):
    
    # RangeK=copy.deepcopy(RangeC)
    
    
    
    
    AttrList=np.array([0,1,2,3,4,5,6,7,8,9,13,14,17,20,
                       21,22,23,25,26,27,28,30,32,33,34,
                       35,36,37,38,39,40,41,42,46,49,50,51,52,
                       53,54,55,56,57,58,59])
    
    RangeK=refineRule(AttrList,RangeK)
  
    # np.random.seed(0)
    # shuffle=np.random.permutation(RangeK.shape[0])
    # RangeE=copy.deepcopy(RangeK[shuffle,:])
    
    RangeE=copy.deepcopy(RangeK)
   
    
    accTremove=rulePrune2(notIdx,RangeE)
    
    print(f'accTremove: {accTremove}')
    
    # if accTremove>=accTec:
        
    #     print(f'k4: {k4}')
    
    #     print(f'accTremove: {accTremove}')