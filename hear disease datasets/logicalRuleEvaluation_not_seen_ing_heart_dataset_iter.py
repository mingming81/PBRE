# Acuire rewards for each epoch, state transition caused by the rules extracted from
# pedagogical rule extraction method. (rule_generation)

# ok
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

import pandas as pd

import sys


from joblib import dump, load

torch.autograd.set_detect_anomaly(True)

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')


# create a data storing path

dirName='./../v3/data2'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='./../v3/data2'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)


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


model=Model(13)
    

# 

model=Model(13)
model.load_state_dict(torch.load(other_data_name+'/heart_nn_classifier.pth'))


X_train=np.load(other_data_name+'/heart_train_x.npy')
Y_train=np.load(other_data_name+'/heart_train_y.npy')
X_test=np.load(other_data_name+'/heart_test_x.npy')
Y_test=np.load(other_data_name+'/heart_test_y.npy')

logicalRules=np.load(other_data_name+'/logical_rules_heart_data_1000_1000.npy',allow_pickle=True)
# print(f"extracted rule shape: {logicalRules.shape}")

# 

# %%
# EDA 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import random

random.seed(0)

data=np.concatenate((X_train,Y_train.reshape(-1,1)),axis=1)
data=np.array(sorted(data, key=lambda x: x[-1]))
dataset=pd.DataFrame(data)



dataset[13]=dataset[13].map({0:'N',1:'A', 2:'B', 3:'C', 4:'D'})

fig, ax = plt.subplots(1, 1,figsize=(4,3))

idx=random.randint(0,33)
idx=random.randint(0,33)
# idx=random.randint(0,33)

print(idx)

g=sns.scatterplot(ax=ax,x=dataset[1],y=dataset[0],alpha=0.5, hue=dataset[13],s=70)
# g.get_legend().remove()

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()

# %%
def getCorrelation(x,arraySet):
    
    data = pd.read_csv(other_data_name+'/heart_clearned_df.csv', sep='\t')
    
    data=data.drop([data.keys()[0]],axis=1)
    
    
    corrmat=data.corr()
    
    corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)
    
    
    arraySet=np.array(sorted(arraySet, key=lambda x: x[0][0,0][0,1]))
    
    lenArr=arraySet.shape[0]
    
    arr=np.full([lenArr,13],None)
    
    for i in range(lenArr):
        
        for j in range(1,14):
            
            arr[i,j-1]=np.array([(arraySet[i,j][0,0][0,0].item()+arraySet[i,j][0,0][0,0].item())/2]).astype(np.float32)[0]
            
    arr=sc.transform(np.concatenate((x,arr),axis=0))
    # arr=np.concatenate((x,arr),axis=0)
    
    arr=arr*corrAtrr
    # arr=arr[:,resIdx[0]]*corrAtrr
    
    
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


importAtt=np.array([i for i in range(14)])



def getSpecies(y_pred,y_actual, x):

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
            
            if j in importAtt:
                
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
            
            else:
                pass_att+=1
                
        
        if pass_att==13: 
            
            
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
                
    
    if num_rules==1:
        
        x_s_rule=copy.deepcopy(idx_toDecide[0,0][0,0][0,1]).reshape((1,-1))
        y_rule=copy.deepcopy(x_s_rule)
        
    elif num_rules>1:
        
        # print('multipleeeeeeeee')
        
        # print(num_rules)
        
        # import sys
        # sys.exit()
    
        idx_toDecide=np.array(idx_toDecide).reshape(-1,14)
        
        idx_toDecide=np.array(sorted(idx_toDecide,key=lambda x: x[0][0,0][0,1]))
        
        classNote=np.full([1,idx_toDecide.shape[0]],0).astype(np.float32)
        
        
        # ******************************************
        # print(idx_toDecide.shape)
        # print('*********')
        
        appranceFre=getConFre(idx_toDecide)
        
        # print(appranceFre)
        
        
        # ******************************************
        
        corr=getCorrelation(x, idx_toDecide)
        
        # print(corr)
        
        # print('*********')
        
        # idx_toDecideImporAtt=idx_toDecide[:, [importAtt+1]]
        
        
        for i in range(corr.shape[0]-1):
            # cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
            classNote[0,i]+=corr[0,i+1]
            
        for i in range(appranceFre.shape[1]):
        #             # cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
            classNote[0,i]+=appranceFre[0,i]
            
        # if corr.shape[0]>2:
            
        #     if abs(corr[0,2]-corr[0,1])<=0.001:
                
        #         print('freqqqqqqqqqqqqqqqqqqq')
                
        #         for i in range(appranceFre.shape[1]):
        #             # cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
        #             classNote[0,i]+=appranceFre[0,i]
            
            
        # ******************************************
        
        # otherCorr=getCorrelation(x,otherRules)
        
        # for i in range(otherCorr.shape[0]-1):
            
        #     cla=np.array(otherRules[i,0][0,0][0,1]).astype('int')  
        #     classNote[0,cla]+=otherCorr[0,i+1]
        
        # ******************************************
        # corrAttrApp=getAttrAppCorre(x, idx_toDecide)
        
        # for i in range(corrAttrApp.shape[0]-1):
        #     cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
        #     classNote[0,cla]+=corrAttrApp[0,i+1]
        
   
        # classNoteSum=sum(classNote[0])
        # classNoteP=(classNote/classNoteSum).astype(np.float32)
        
        idx=np.argmax(classNote[0])
        
        
        
        y_rule=idx_toDecide[idx,0][0,0][0,1]
        
        
        
        # print(y_pred, x_s_rule)
        
        # if y_pred==x_s_rule:
            
        #     print('here')
            
        #     y_rule=copy.deepcopy(x_s_rule) 
            
        #     return y_rule
        
        # else:
      
        #     idxx=toDeleteIdx[0,idx]
        #     logicalRules[idxx,0][0,1]-=10
            
        #     x_s_rule=getSpecies(y_pred,acc_rule,x)
        
    # if y_rule!=y_pred:
    #     sys.exit()
        
        
            
            
    return y_rule

# 
lenX=X_train.shape[0]
# lenX=X_test.shape[0]

# %%
# B
max_acc=0

data = pd.read_csv(other_data_name+'/heart_clearned_df.csv', sep='\t')

data=data.drop([data.keys()[0]],axis=1)

corrmat=data.corr()

corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)

indexes=np.array(sorted(corrAtrr.reshape(-1,1), key=lambda x: abs(x)))


notsignificant=np.full([1],None)

notsignificantcp2=copy.deepcopy(notsignificant)

# %%
# F

# max_acc=np.load(other_data_name+'/heart_test_acc_max.npy', allow_pickle=(True))  
# notsignificant=np.load(other_data_name+'/heart_not_signigicant_B.npy',allow_pickle=(True)).astype('float32')  

# notsignificantcp=copy.deepcopy(notsignificant)
# notsignificantcp2=copy.deepcopy(notsignificant)

# %%
# Check
# notsignificant=np.load(other_data_name+'/heart_not_signigicant_F.npy', allow_pickle=(True))


# %%
    
# B
for idx_todecide in indexes:
    
    idx=np.where(corrAtrr==idx_todecide[0])[1][0]
    
    
    if notsignificant[0]==None:
        
        notsignificant=np.array([idx])
        
    else:
        notsignificant=np.concatenate((notsignificant, np.array([idx])))
    
    
# # F   
# for idx_todecide in range(notsignificantcp.shape[0]):
    
#     notsignificant[idx_todecide]=np.NaN
    
# Check
# for idx_todecide in range(1):   
    
    
    
    sc=load(dirName+'/heart_std_scaler.bin')
    num_similarity=np.zeros((1,1))
    num_proposable=np.zeros((1,1))
    num_acc_numeric=np.zeros((1,1))
    num_acc_rule=np.zeros((1,1))
    acc_rule=0
    
    
    for i in range(lenX):
        
        # print('===========================')
      
        # print(f'step: {i}')
        
        # if i==44:
        #     print('yes')
        
        x=X_train[[i]]
        # x=X_test[[i]]
        
        x=torch.from_numpy(x).type(torch.float32)
        x_rule=copy.deepcopy(x)
        
        x=sc.transform(x)
    
        x=torch.from_numpy(x).type(torch.float32)
        
        y_actual=torch.from_numpy(Y_train[[i]]).type(torch.float32)
        # y_actual=torch.from_numpy(Y_test[[i]]).type(torch.float32)
        
        
            
        y_pred=model(x).reshape((1,-1))
        
        y_pred=torch.argmax(y_pred, dim=1).reshape((1,-1))
    
        y_rule=getSpecies(y_pred,y_actual,x_rule)
        
        if y_rule!=None:
            y_rule=y_rule.reshape((1,-1))
        
        # print(f'y_pred: {(y_pred)}, y_rule: {y_rule}, y_true: {y_actual}')
    
        if y_rule!=None:
            if y_pred.item()==y_rule.item():
                num_similarity[0,0]+=1
                
            else:
                pass
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
             
        
        if i!=0:
            acc_rule=num_acc_rule/(i+1)
            
    if (num_acc_rule/lenX)>=max_acc:
        
        notsignificantcp2=copy.deepcopy(notsignificant)
        
        max_acc=copy.deepcopy(num_acc_rule/lenX)
        
    else:
        notsignificant=copy.deepcopy(notsignificantcp2)
        
        
    print(f'notsignificant: {notsignificant}')   

    print(f'max_acc: {max_acc}')  
     
        
np.save(other_data_name+'/heart_test_acc_max.npy',max_acc)  
np.save(other_data_name+'/heart_not_signigicant_B.npy',notsignificant)  
  
np.save(other_data_name+'/heart_not_signigicant_F.npy',notsignificant)    
                
            
  

print(f'num_similarity: {num_similarity}/{lenX}')
print(f'num_proposable: {num_proposable}/{lenX}')
print(f'num_acc_numeric: {num_acc_numeric}/{lenX}')
print(f'num_acc_rule: {num_acc_rule}/{lenX}')

print(f'num_similarity: {num_similarity/lenX}')
print(f'num_proposable: {num_proposable/lenX}')
print(f'num_acc_numeric: {num_acc_numeric/lenX}')
print(f'num_acc_rule: {num_acc_rule/lenX}')

np.save(other_data_name+'/irir_similarity.npy',num_similarity)
np.save(other_data_name+'/irir_predictability.npy',num_proposable)
np.save(other_data_name+'/iris_reward_numeric.npy',num_acc_numeric)
np.save(other_data_name+'/iris_reward_rule.npy',num_acc_rule)


