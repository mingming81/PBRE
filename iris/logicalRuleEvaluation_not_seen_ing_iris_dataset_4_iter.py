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

import pandas as pd

import sys


from joblib import dump, load

torch.autograd.set_detect_anomaly(True)

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
    



model=Model(4)
model.load_state_dict(torch.load(other_data_name+'/iris_nn_classifier.pth'))

# X_train=np.load(other_data_name+'/iris_trainsetX.npy',allow_pickle=True)
# Y_train=np.load(other_data_name+'/iris_trainsetY.npy',allow_pickle=True)
# X_test=np.load(other_data_name+'/iris_testsetX.npy',allow_pickle=True)
# Y_test=np.load(other_data_name+'/iris_testsetY.npy',allow_pickle=True)

X_train=np.load(other_data_name+'/iris_trainsetX_4.npy',allow_pickle=True)
Y_train=np.load(other_data_name+'/iris_trainsetY_4.npy',allow_pickle=True)
X_test=np.load(other_data_name+'/iris_testsetX_4.npy',allow_pickle=True)
Y_test=np.load(other_data_name+'/iris_testsetY_4.npy',allow_pickle=True)



# %%
# EDA 

X_train=np.load(other_data_name+'/iris_trainsetX_4.npy')
Y_train=np.load(other_data_name+'/iris_trainsetY_4.npy')
X_test=np.load(other_data_name+'/iris_testsetX_4.npy')
Y_test=np.load(other_data_name+'/iris_testsetY_4.npy')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

X=np.concatenate((X_train, X_test),axis=0)
Y=np.concatenate((Y_train, Y_test))
data=np.concatenate((X_train,Y_train.reshape(-1,1)),axis=1)
data=np.array(sorted(data, key=lambda x: x[-1]))
dataset=pd.DataFrame(data)

dataset[4]=dataset[4].map({0:'Iris Setosa',1: 'Iris Versicolour', 2:'Iris Virginica'})

fig, ax = plt.subplots(1, 1,figsize=(4,3))

g=sns.scatterplot(ax=ax,x=dataset[3],y=dataset[0],alpha=0.6, hue=dataset[4],s=70)
# g.get_legend().remove()

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()



# %%

# X_train=np.load(other_data_name+'/iris_trainsetX_5.npy')
# Y_train=np.load(other_data_name+'/iris_trainsetY_5.npy')
# X_test=np.load(other_data_name+'/iris_testsetX_5.npy')
# Y_test=np.load(other_data_name+'/iris_testsetY_5.npy')
# X_val=np.load(other_data_name+'/iris_valsetX_5.npy')
# Y_val=np.load(other_data_name+'/iris_valsetY_5.npy')

logicalRules=np.load(other_data_name+'/logical_rules_iris_data_5.npy',allow_pickle=True)
# print(f"extracted rule shape: {logicalRules.shape}")




def getCorrelation(x,arraySet):
    
    data = pd.read_csv(other_data_name+'/Iris.csv')


    data = data.drop(['Id'], axis=1)
    
    d={'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    
    data['Species']=data['Species'].map(d)
    
    corrmat=data.corr()
    
    corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)
    
    
    arraySet=np.array(sorted(arraySet, key=lambda x: x[0][0,0][0,1]))
    
    lenArr=arraySet.shape[0]
    
    arr=np.full([lenArr,4],None)
    
    for i in range(lenArr):
        for j in range(1,5):
            
            arr[i,j-1]=np.array([(arraySet[i,j][0,0][0,0].item()+arraySet[i,j][0,0][0,0].item())/2]).astype(np.float32)[0]
            
    arr=np.concatenate((x,arr),axis=0)*(corrAtrr)
    
    arr=arr.astype(np.float32)
    
    if notsignificant[0]!=None:
        
        notsignificantcp3=np.array([i for i in notsignificant if not np.isnan(i)])
        
        if notsignificantcp3.shape[0]!=0:
    
            arr=np.delete(arr,notsignificantcp3.astype('int'),axis=1)
    
    
    # print(arr.shape)
    
    corr=np.corrcoef(arr)
    
    # return corr
    
    
    
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


importAtt=np.array([0,1,2,3,4])



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
                
                
                if j in notsignificant.astype('int'):
                    pass_att+=1
                    continue
                
                if x[0,j]>=logicalRules[i,j+1][0,0][0,1] and x[0,j]<=logicalRules[i,j+1][0,0][0,2]:
                    
                    pass_att+=1
                    
                else:
                    break
            
            else:
                pass_att+=1
                
        
        if pass_att==4: 
            
            
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
        
        # print('multiple')
        
        # print(num_rules)
        
        # import sys
        # sys.exit()
    
        idx_toDecide=np.array(idx_toDecide).reshape(-1,5)
        
        classNote=np.full([1,3],0).astype(np.float32)
        
        
        # ******************************************
        
        appranceFre=getConFre(idx_toDecide)
        
        
        # ******************************************
        
        corr=getCorrelation(x, idx_toDecide)
        
        # print(f'====================corr: {corr}')
        
        # idx_toDecideImporAtt=idx_toDecide[:, [importAtt+1]]
        
        
        for i in range(corr.shape[0]-1):
            cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
            classNote[0,cla]+=corr[0,i+1]
            
        if abs(corr[0,2]-corr[0,1])<=0.0001:
            
            # print('freq')
            
            for i in range(appranceFre.shape[1]):
                cla=np.array(idx_toDecide[i,0][0,0][0,1]).astype('int')
                classNote[0,cla]+=appranceFre[0,i]
            
            
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
        
    # if y_rule!=y_pred:
    #     sys.exit()
        
        
            
            
    return y_rule

# %%
# B
max_acc=0

data = pd.read_csv(other_data_name+'/Iris.csv')


data = data.drop(['Id'], axis=1)

d={'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

data['Species']=data['Species'].map(d)

corrmat=data.corr()

corrAtrr=np.array(corrmat.iloc[:-1,[-1]]).reshape(1,-1)

indexes=np.array(sorted(corrAtrr.reshape(-1,1), key=lambda x: abs(x)))

notsignificant=np.full([1],None)

notsignificantcp2=copy.deepcopy(notsignificant)

# %%
# F
max_acc=np.load(other_data_name+'/iris_test_acc_max.npy')  
notsignificant=np.load(other_data_name+'/iris_not_signigicant_B.npy').astype('float32')  

notsignificantcp=copy.deepcopy(notsignificant)
notsignificantcp2=copy.deepcopy(notsignificant)

# %%
notsignificant=np.load(other_data_name+'/iris_not_signigicant_F.npy').astype('float32')  

# %%
# # B
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
    
    
    
    # lenX=X_train.shape[0]
    lenX=X_test.shape[0]
    # lenX=X_val.shape[0]
    
    num_similarity=np.zeros((1,1))
    num_proposable=np.zeros((1,1))
    num_acc_numeric=np.zeros((1,1))
    num_acc_rule=np.zeros((1,1))
    acc_rule=0
    
    sc=load(dirName+'/iris_nn_std_scaler.bin')
    
    
    for i in range(lenX):
      
        # print(f'step: {i}')
        
        # if i==44:
        #     print('yes')
        
        # x=X_train[[i]]
        x=X_test[[i]]
        # x=X_val[[i]]
        
        x=torch.from_numpy(x).type(torch.float32)
        x_rule=copy.deepcopy(x)
        
        x=sc.transform(x)
    
        x=torch.from_numpy(x).type(torch.float32)
        
        # y_actual=torch.from_numpy(Y_train[[i]]).type(torch.float32)
        y_actual=torch.from_numpy(Y_test[[i]]).type(torch.float32)
        # y_actual=torch.from_numpy(Y_val[[i]]).type(torch.float32)
        
        
            
        y_pred=model(x).reshape((1,-1))
        
        y_pred=torch.argmax(y_pred, dim=1).reshape((1,-1))
    
        y_rule=getSpecies(y_pred,y_actual,x_rule)
        
        if y_rule!=None:
            y_rule=y_rule.reshape((1,-1))
        
        # print(f'y_pred: {(y_pred)}, y_rule: {y_rule}')
    
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
            
            
    # if (num_acc_rule/lenX)>=max_acc:
        
    #     notsignificantcp2=copy.deepcopy(notsignificant)
        
    #     max_acc=copy.deepcopy(num_acc_rule/lenX)
        
    # else:
    #     notsignificant=copy.deepcopy(notsignificantcp2)
    
    
    
        
    print(f'notsignificant: {notsignificant}')   

    print(f'max_acc: {max_acc}')  
     
        
# np.save(other_data_name+'/iris_test_acc_max.npy',max_acc)  
# np.save(other_data_name+'/iris_not_signigicant_B.npy',notsignificant)   
 
# np.save(other_data_name+'/iris_not_signigicant_F.npy',notsignificant)    
            
      

print(f'num_similarity: {num_similarity}/{lenX}')
print(f'num_proposable: {num_proposable}/{lenX}')
print(f'num_acc_numeric: {num_acc_numeric}/{lenX}')
print(f'num_acc_rule: {num_acc_rule}/{lenX}')
print('===========================')
print(f'num_similarity: {num_similarity/lenX}')
print(f'num_proposable: {num_proposable/lenX}')
print(f'num_acc_numeric: {num_acc_numeric/lenX}')
print(f'num_acc_rule: {num_acc_rule/lenX}')

np.save(other_data_name+'/irir_similarity.npy',num_similarity)
np.save(other_data_name+'/irir_predictability.npy',num_proposable)
np.save(other_data_name+'/iris_reward_numeric.npy',num_acc_numeric)
np.save(other_data_name+'/iris_reward_rule.npy',num_acc_rule)


