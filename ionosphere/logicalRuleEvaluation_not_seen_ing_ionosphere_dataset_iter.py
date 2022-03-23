# Acuire rewards for each epoch, state transition caused by the rules extracted from
# pedagogical rule extraction method. (rule_generation)
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
    
# %%

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        num_input_features=33

       
       
       
        num_output_features=1

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(num_input_features,34),
            torch.nn.ReLU(),
            # torch.nn.Linear(60,60),
            # torch.nn.ReLU(),
            # torch.nn.Linear(60,10),
            # torch.nn.ReLU(),
            torch.nn.Linear(34,num_output_features),
            torch.nn.Sigmoid()

            )

    def forward(self,x):
        x = self.flatten(x)
        y=self.linear_relu_stack(x)

        return y


model=Model()

model.load_state_dict(torch.load(other_data_name+'/inosphere_nn_classifier.pth'))





logicalRules=np.load(other_data_name+'/logical_rules_inosphere_data.npy',allow_pickle=True)
print(f"extracted rule shape: {logicalRules.shape}")


# dataset = pd.read_csv('./v3/data2/ionosphere.data.csv', header=None)

X=np.load(other_data_name+'/ionosphere_x.npy')
y=np.load(other_data_name+'/ionosphere_y.npy')




X_train = np.load(other_data_name+'/ionosphere_x_train.npy',allow_pickle=True)


y_train = np.load(other_data_name+'/ionosphere_y_train.npy',allow_pickle=True)

X_test  = np.load(other_data_name+'/ionosphere_x_test.npy',allow_pickle=True)


y_test  = np.load(other_data_name+'/ionosphere_y_test.npy',allow_pickle=True)

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



dataset[33]=dataset[33].map({0:'B',1: 'G'})

fig, ax = plt.subplots(1, 1,figsize=(4,3))

idx=random.randint(0,33)
idx=random.randint(0,33)
# idx=random.randint(0,33)

print(idx)

g=sns.scatterplot(ax=ax,x=dataset[31],y=dataset[0],alpha=0.5, hue=dataset[33],s=70)
g.get_legend().remove()

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()

# %%



def getCorrelation(x, arraySet):
    
    
    data=np.concatenate((X,y.reshape(-1,1)),axis=1)
    
    dataset=pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    
    # data=pd.DataFrame(data=dataset)
    
    
    
    corrmat=dataset.corr()
    
    corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)
    
    arraySet=(np.array(sorted(arraySet, key=lambda x: x[0][0,0][0,1])))
    
    # arraySet=sc.transform(arraySet)
    
    lenArr=arraySet.shape[0]
    arr=np.full([lenArr,33],None)
    
    x=copy.deepcopy(np.array(x).astype(np.float32))
    
    
    
    for i in range(lenArr):
        
        for j in range(1,34):
            
            arr[i,j-1]=np.array([(arraySet[i,j][0,0][0,0].item()+arraySet[i,j][0,0][0,0].item())/2]).astype(np.float32)[0]
            
    
    
    arr=np.concatenate((x,arr),axis=0)*(corrAtrr)
   
    arr=arr.astype(np.float32)
    
    if notsignificant[0]!=None:
        
        notsignificantcp3=np.array([i for i in notsignificant if not np.isnan(i)])
        
        if notsignificantcp3.shape[0]!=0:
    
            arr=np.delete(arr,notsignificantcp3.astype('int'),axis=1)
    
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
    
        idx_toDecide=np.array(idx_toDecide).reshape(-1,34)
        
        
        
        classNote=np.full([1,2],0).astype(np.float32)
        
        
        # ******************************************
        
        appranceFre=getConFre(copy.deepcopy(idx_toDecide))
        
        # print(f'appranceFre: {appranceFre}')
        
        # for i in range(appranceFre.shape[1]):
        #     cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
        #     classNote[0,cla]+=appranceFre[0,i]
            
            
        # ******************************************
        
        corr=getCorrelation(x, copy.deepcopy(idx_toDecide))
        
        # print(f'corr: {corr}')
        
        for i in range(corr.shape[0]-1):
            cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
            classNote[0,cla]+=abs(corr[0,i+1])
            
        if abs(corr[0,2]-corr[0,1])<1:
            
            # print('freqqqqqqqqqqq')
            
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

# %%

# B
data=np.concatenate((X,y.reshape(-1,1)),axis=1)

dataset=pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])

corrmat=dataset.corr()

corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)

indexes=np.array(sorted(corrAtrr.reshape(-1,1), key=lambda x: abs(x)))

notsignificant=np.full([1],None)

notsignificantcp2=copy.deepcopy(notsignificant)

max_acc=0

# %%
# F
max_acc=np.load(other_data_name+'/ionosphere_test_acc_max.npy')  
notsignificant=np.load(other_data_name+'/ionosphere_not_signigicant_B.npy').astype('float32')  

notsignificantcp=copy.deepcopy(notsignificant)
notsignificantcp2=copy.deepcopy(notsignificant)

# %%

# Check

notsignificant=np.load(other_data_name+'/ionosphere_not_signigicant_F.npy',allow_pickle=(True)).astype('float32')  




# %%

# B
# for idx_todecide in indexes:
    
#     idx=np.where(corrAtrr==idx_todecide[0])[1][0]
    
#     if notsignificant[0]==None:
        
#         notsignificant=np.array([idx])
        
#     else:
#         notsignificant=np.concatenate((notsignificant, np.array([idx])))
    
    
# F
# for idx_todecide in range(notsignificantcp.shape[0]):
    
#     notsignificant[idx_todecide]=np.NaN
    
    
# Check
for idx_todecide in range(1):
    
    
    
    
        
    
    
    
    
    num_similarity=np.zeros((1,1))
    num_proposable=np.zeros((1,1))
    num_acc_numeric=np.zeros((1,1))
    num_acc_rule=np.zeros((1,1))
    acc_rule=0
  
    
    lenX=X_train.shape[0]
    # lenX=X_test.shape[0]

    
    for i in range(lenX):
      
        # print(f'step: {i}')
        
        # if i==2:
        #     print('yes')
        
        x=X_train[[i]]
        # x=X_test[[i]]
    
        x=torch.from_numpy(x).type(torch.float32)
        
        x_rule=copy.deepcopy(x)
        
     
        y_actual=torch.from_numpy(y_train[[i]]).type(torch.float32)
        # y_actual=torch.from_numpy(y_test[[i]]).type(torch.float32)
    
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
            
            
            
    # if (num_acc_rule/lenX)>=max_acc:
        
    #     notsignificantcp2=copy.deepcopy(notsignificant)
        
    #     max_acc=copy.deepcopy(num_acc_rule/lenX)
        
    # else:
    #     notsignificant=copy.deepcopy(notsignificantcp2)
        
        
        
        
        
        
        
    print(f'notsignificant: {notsignificantcp2}')   

    print(f'max_acc: {max_acc}')
    # 
    print('========================')
     
        
# np.save(other_data_name+'/ionosphere_test_acc_max.npy',max_acc)  
# np.save(other_data_name+'/ionosphere_not_signigicant_B.npy',notsignificant)    




# np.save(other_data_name+'/ionosphere_not_signigicant_F.npy',notsignificantcp2)    
                
        
    

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