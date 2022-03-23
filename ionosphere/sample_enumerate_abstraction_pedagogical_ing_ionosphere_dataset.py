

import torch 
import numpy as np

from tree_ing_ionosphere_dataset import *

from joblib import dump, load


import os

# %%

dirName='./../v3/data2'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='./../v3/data2'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)
    


# X: torch tensor 1*n
def one_hot_decoder(X):
    idx=(X == 1).nonzero(as_tuple=False)
    num=idx[0,1].reshape((1,1))
    return num


def digi_str(x_s):
    x_s_str=np.full([1,2], None)

    x_s_str[0,0]= x_s
    x_s_str[0,1]=x_s

    return x_s_str
    
    
# seapeda: sample enumerate abstraction pedagogical method
class Seapeda:
    def __init__(self):
        self.ruleTree=Node('root')
    
    def ruleExtract(self,x,model):
        
        
            
        self.rule=np.full([1,34],None)
        
        x_original=copy.deepcopy(x)

        x_original=torch.from_numpy(x_original).type(torch.float32)
        
        x=torch.from_numpy(x).type(torch.float32)
        
        out=model(x)

        out=out.reshape(-1).detach().numpy().astype(np.float32).round()

        out=torch.from_numpy(out).type(torch.float32)
        

        x_o=out.reshape((1,-1))
        
        self.rule[0,0]=digi_str(x_o)
        
        for i in range(33):
            x_tr=digi_str(x_original[0,i].reshape((1,-1)))
            self.rule[0,i+1]=x_tr
            
        return self.rule
        
        

    
    def update(self,rule):
        
        self.ruleTree.update(toCheck=rule)

    def pruneNodes(self):
        self.ruleTree.pruneNodes()
        
    def refineRule(self):
        fullSet=self.ruleTree.refineRule()
        return fullSet
    

        
    
    

# %%

# generate testing dataset



# %%
# seapeda=Seapeda()
# X=np.full([100,6],None)

# for x in X:
    
#     rule=seapeda.ruleExtract(x,model)

#     seapeda.update(rule)
    
    
#     rule=seapeda.refineRule()
    
    
    





    
        
    
        
            