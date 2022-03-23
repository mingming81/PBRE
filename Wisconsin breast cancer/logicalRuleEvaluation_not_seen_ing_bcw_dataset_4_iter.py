# Acuire rewards for each epoch, state transition caused by the rules extracted from
# pedagogical rule extraction method. (rule_generation)

# rule_generation_not_seein_bcw_2.py ok
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

from collections import deque
import random
import copy
import math
import os
import pickle

import math

import numpy as np

from tensorflow.keras import models, layers, optimizers, callbacks

import pandas as pd 

from sklearn.model_selection import train_test_split

torch.autograd.set_detect_anomaly(True)

from joblib import dump, load

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')

# %%
# create a data storing path

dirName='./../v3/data2'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='./../v3/data2'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)
    


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



model=ServiceModel()

model.load_state_dict(torch.load(other_data_name+'/bcw_nn_classifier.pth'))



logicalRules=np.load(other_data_name+'/logical_rules_bcw_data_2.npy',allow_pickle=True)
# print(f"extracted rule shape: {logicalRules.shape}")


sc=load(other_data_name+'/bcw_std_scaler.bin')

X_train = np.load(other_data_name+'/bcw_x_train.npy',allow_pickle=True)
y_train = np.load(other_data_name+'/bcw_y_train.npy',allow_pickle=True)

X_test  = np.load(other_data_name+'/bcw_x_test.npy',allow_pickle=True)

y_test  = np.load(other_data_name+'/bcw_y_test.npy',allow_pickle=True)


# %%
# EDA 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import random

random.seed(0)

data=np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
data=np.array(sorted(data, key=lambda x: x[-1]))
dataset=pd.DataFrame(data)



dataset[30]=dataset[30].map({0:'B',1: 'M'})

fig, ax = plt.subplots(1, 1,figsize=(4,3))

idx=random.randint(0,29)

print(idx)

g=sns.scatterplot(ax=ax,x=dataset[14],y=dataset[0],alpha=0.6, hue=dataset[30],s=70)
# g.get_legend().remove()

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()



# %%



# X_test, X_val, Y_test, Y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

# y_test=copy.deepcopy(Y_test)

# np.save(other_data_name+'/bcw_x_train_5.npy', X_train)
# np.save(other_data_name+'/bcw_y_train_5.npy', y_train)
# np.save(other_data_name+'/bcw_x_test_5.npy', X_test)
# np.save(other_data_name+'/bcw_y_test_5.npy', Y_test)
# np.save(other_data_name+'/bcw_x_val_5', X_val)
# np.save(other_data_name+'/bcw_y_val_5.npy', Y_val)

# X_train=np.load(other_data_name+'/bcw_x_train_5.npy')
# Y_train=np.load(other_data_name+'/bcw_y_train_5.npy')
# X_test=np.load(other_data_name+'/bcw_x_test_5.npy')
# Y_test=np.load(other_data_name+'/bcw_y_test_5.npy')
# X_val=np.load(other_data_name+'/bcw_x_val_5')
# Y_val=np.load(other_data_name+'/bcw_y_val_5.npy')


# y_test=Y_test


def getCorrelation(x, arraySet):
    
    X=np.concatenate((X_train,X_test),axis=0)
    y=np.concatenate((y_train,y_test))
    
    data=np.concatenate((X,y.reshape(-1,1)),axis=1)
    
    dataset=pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    
    data=pd.DataFrame(data=dataset)
    
    corrmat=data.corr()
    
    corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)
    
    arraySet=np.array(sorted(arraySet, key=lambda x: x[0][0,0][0,1]))
    
    lenArr=arraySet.shape[0]
    arr=np.full([lenArr,30],None)
    
    x=copy.deepcopy(np.array(x).astype(np.float32))
    
    
    for i in range(lenArr):
        for j in range(1,31):
            arr[i,j-1]=np.array([(arraySet[i,j][0,0][0,0].item()+arraySet[i,j][0,0][0,0].item())/2]).astype(np.float32)[0]
            
    
    
    arr=np.concatenate((x,arr),axis=0)*(corrAtrr)
    
    arr=arr.astype(np.float32)
    
    if notsignificant[0]!=None:
        
        notsignificantcp3=np.array([i for i in notsignificant if not np.isnan(i)])
        
        if notsignificantcp3.shape[0]!=0:
    
            arr=np.delete(arr,notsignificantcp3.astype('int'),axis=1)
    
    
    # print(arr.shape)
    
    corr=np.corrcoef(arr)
    
    return corr


            
            
def getConFre(arraySet):
    
    arraySet=np.array(sorted(arraySet, key=lambda x: x[0][0,0][0,1]))
    
    lenArr=arraySet.shape[0]
    arr=np.full([1,lenArr],None)
    for i in range(lenArr):
        arr[0,i]=arraySet[i,0][0,1]
    
    sumarr=sum(arr[0])
    
    arr=(arr/sumarr).astype(np.float32)
    
    return arr
    
    
    
            
    
    

def getSpecies(y_pred,acc_rule, x):

    lenRules=logicalRules.shape[0]  
    
    otherRules=copy.deepcopy(logicalRules)

    x_s_rule=None
    
    y_rule=None

    idx_toDecide=np.full([999,999],None)
    
    toDeleteIdx=np.array([[None]])
    
    num_rules=0
        
    for i in range(lenRules):
        
        pass_att=0
        
        for j in range(0,logicalRules.shape[1]-1):
            
            if notsignificant[0]!=None:
                
                notsignificantcp3=np.array([i for i in notsignificant if not np.isnan(i)])
                
                if notsignificantcp3.shape[0]!=0:
                    
                    if j in notsignificant.astype('int'):
                        pass_att+=1
                        continue
            
            
            if x[0,j]>=logicalRules[i,j+1][0,0][0,1] and x[0,j]<=logicalRules[i,j+1][0,0][0,2]:
                
                pass_att+=1
                
            else:
                break
                

        if pass_att==(logicalRules.shape[1]-1): 
            
            
            num_rules+=1
            
            if (idx_toDecide.shape==(999,999)):
                idx_toDecide=copy.deepcopy(logicalRules[[i]])
                
    
            else:
                
                idx_toDecideInt=None
                idx_toDecideInt=copy.deepcopy(logicalRules[[i]])
                idx_toDecide=np.concatenate((idx_toDecide,idx_toDecideInt),axis=0)
                
                
            if toDeleteIdx[0,0]==None:
                toDeleteIdx[0,0]=i
            else:
                toDeleteIdx=np.concatenate((toDeleteIdx, np.array([[i]])),axis=1)
                
    # otherRules=np.delete(otherRules,toDeleteIdx.astype('int'),axis=0)
   
    
    if num_rules==1:
        x_s_rule=copy.deepcopy(idx_toDecide[0,0][0,0][0,1]).reshape((1,-1))
        y_rule=copy.deepcopy(x_s_rule)
        
    elif num_rules>1:
        
        # print('multiple')
        
        # print(num_rules)
    
        idx_toDecide=np.array(idx_toDecide).reshape(-1,31)
        
        
        
        classNote=np.full([1,2],0).astype(np.float32)
        
        
        # ******************************************
        
        appranceFre=getConFre(idx_toDecide)
        
        # for i in range(appranceFre.shape[1]):
        #     cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
        #     classNote[0,cla]+=appranceFre[0,i]
            
            
        # ******************************************
        
        corr=getCorrelation(x, idx_toDecide)
        
        for i in range(corr.shape[0]-1):
            cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
            classNote[0,cla]+=corr[0,i+1]
            
        if abs(corr[0,2]-corr[0,1])<=0.0001:
            
            # print('freq')
            
            for i in range(appranceFre.shape[1]):
                cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
                classNote[0,cla]+=appranceFre[0,i]
        
        # ******************************************
        # corrAttrApp=getAttrAppCorre(x, idx_toDecide)
        
        # for i in range(corrAttrApp.shape[0]-1):
        #     cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
        #     classNote[0,cla]+=corrAttrApp[0,i+1]*(2/5)
        # ******************************************
        
        # otherCorr=getCorrelation(x,otherRules)
        
        # for i in range(otherCorr.shape[0]-1):
            
        #     cla=np.array(otherRules[i,0][0,0][0,1]).astype('int')  
        #     classNote[0,cla]+=otherCorr[0,i+1]
   
        
        
        idx=np.argmax(classNote[0])
        
        y_rule=torch.tensor([[idx]]).type(torch.float32)
        
        
        
        # print(y_pred, x_s_rule)
        
        # if y_pred==x_s_rule:
            
        #     print('here')
            
        #     y_rule=copy.deepcopy(x_s_rule) 
            
        #     return y_rule
        
        # else:
      
        #     idxx=toDeleteIdx[0,idx]
        #     logicalRules[idxx,0][0,1]-=10
            
        #     x_s_rule=getSpecies(y_pred,acc_rule,x)
            
            
    return y_rule




lenX=X_train.shape[0]
# lenX=X_test.shape[0]
# lenX=X_val.shape[0]

# %%
# B
max_acc=0

X=np.concatenate((X_train,X_test),axis=0)
y=np.concatenate((y_train,y_test))

data=np.concatenate((X,y.reshape(-1,1)),axis=1)

dataset=pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])

data=pd.DataFrame(data=dataset)

corrmat=data.corr()

corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)

indexes=np.array(sorted(corrAtrr.reshape(-1,1), key=lambda x: abs(x)))


notsignificant=np.full([1],None)

notsignificantcp2=copy.deepcopy(notsignificant)

# %%
# F

max_acc=np.load(other_data_name+'/bcw_test_acc_max.npy')  
notsignificant=np.load(other_data_name+'/bcw_not_signigicant_B.npy').astype('float32')  

notsignificantcp=copy.deepcopy(notsignificant)
notsignificantcp2=copy.deepcopy(notsignificant)

# %%
# check
notsignificant=np.load(other_data_name+'/bcw_not_signigicant_F.npy').astype('float32') 

# %%
# B
for idx_todecide in indexes:
    
    idx=np.where(corrAtrr==idx_todecide[0])[1][0]
    
    # idx=idx_todecide
    
    if notsignificant[0]==None:
        
        notsignificant=np.array([idx])
        
    else:
        notsignificant=np.concatenate((notsignificant, np.array([idx])))
    
    
# F   
# for idx_todecide in range(notsignificantcp.shape[0]):
#     notsignificant[idx_todecide]=np.NaN

# Check
# for idx_todecide in range(1):

    
    
    
    num_similarity=np.zeros((1,1))
    num_proposable=np.zeros((1,1))
    num_acc_numeric=np.zeros((1,1))
    num_acc_rule=np.zeros((1,1))
    
    acc_rule=0
    
    for i in range(lenX):
      
        # print(f'step: {i}')
        
        
        x=X_train[[i]]
        # x=X_test[[i]]
        # x=X_val[[i]]
    
        x=torch.from_numpy(x).type(torch.float32)
        
        x_rule=copy.deepcopy(x)
        
        x=sc.transform(x)
    
        x=torch.from_numpy(x).type(torch.float32)
        
        y_actual=torch.from_numpy(y_train[[i]]).type(torch.float32)
        # y_actual=torch.from_numpy(y_test[[i]]).type(torch.float32)
        # y_actual=torch.from_numpy(Y_val[[i]]).type(torch.float32)
    
        y_pred=model(x)
    
        y_pred=y_pred.reshape(-1,1).detach().numpy().astype(np.float32).round()
    
        y_pred=torch.from_numpy(y_pred).type(torch.float32)
        
        
    
        y_rule=getSpecies(y_pred,y_actual,x_rule)
        
        if y_rule!=None:
            y_rule=y_rule.reshape((1,-1))
            
        
        # print(f'y_pred: {(y_pred)}, y_rule: {y_rule}')
        
        # if y_rule==None:
        #     print(x_rule)
            # import sys
            # sys.exit()
        
        
        # if y_rule==None:
        #     print('yes')
        
        if y_rule!=None:
            if y_pred.item()==y_rule.item():
                num_similarity[0,0]+=1
                
            else:
                pass
                
                # import sys
                # sys.exit()
            
        else:
            pass
            # print('no')
    
        if y_rule!=None:
            num_proposable[0,0]+=1
    
        if y_pred.item()==y_actual.item():
            num_acc_numeric[0,0]+=1
        
        if y_rule!=None:
            if y_rule.item()==y_actual.item():
                num_acc_rule[0,0]+=1
                
            else:
                pass
                # import sys
                # sys.exit()
                
        
        if i!=0:
            acc_rule=num_acc_rule/(i+1)
            
    
    if (num_acc_rule/lenX)>=max_acc:
        
        notsignificantcp2=copy.deepcopy(notsignificant)
        
        max_acc=copy.deepcopy(num_acc_rule/lenX)
        
    else:
        notsignificant=copy.deepcopy(notsignificantcp2)
        
        
    print(f'notsignificant: {notsignificant}')   

    print(f'max_acc: {max_acc}')  
     
        
np.save(other_data_name+'/bcw_test_acc_max.npy',max_acc)  
np.save(other_data_name+'/bcw_not_signigicant_B.npy',notsignificant)    

# np.save(other_data_name+'/bcw_not_signigicant_F.npy',notsignificant)    
                
        


print(f'num_similarity: {num_similarity}/{lenX}')
print(f'num_proposable: {num_proposable}/{lenX}')
print(f'num_acc_numeric: {num_acc_numeric}/{lenX}')
print(f'num_acc_rule: {num_acc_rule}/{lenX}')

print(f'=======================================')

print(f'num_similarity: {num_similarity/lenX}')
print(f'num_proposable: {num_proposable/lenX}')
print(f'num_acc_numeric: {num_acc_numeric/lenX}')
print(f'num_acc_rule: {num_acc_rule/lenX}')

np.save(other_data_name+'/eca_similarity.npy',num_similarity)
np.save(other_data_name+'/eca_predictability.npy',num_proposable)
np.save(other_data_name+'/eca_reward_numeric.npy',num_acc_numeric)
np.save(other_data_name+'/eca_reward_rule.npy',num_acc_rule)


# np.save(other_data_name+'/logical_rules_bcw_data_3.npy',logicalRules)