import torch 
import numpy as np

from tree_ing_heart_dataset_2 import *

from joblib import dump, load

max_x_light=1005
# %%
X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)

dirName='./../v3/data2'

sc=load(dirName+'/heart_std_scaler.bin')

# X: torch tensor 1*n
def one_hot_decoder(X):
    idx=(X == 1).nonzero(as_tuple=False)
    num=idx[0,1].reshape((1,1))
    return num

def digi_str(x_s):

    x_s_str=np.full([1,2], None)

    x_s_0=np.full([1,2], None)
    x_s_0[0,0]=x_s  #mean
    x_s_0[0,1]=0   # variance

    x_s_str[0,0]= x_s_0
    x_s_str[0,1]=x_s

    return x_s_str
    
    
# seapeda: sample enumerate abstraction pedagogical method
class Seapeda:
    def __init__(self):
        self.ruleTree=Node('root')
    
    def ruleExtract(self,x,model):
        
        
            
        self.rule=np.full([1,14],None)

        x=torch.from_numpy(x).type(torch.float32)
        x_original=copy.deepcopy(x).reshape((1,-1))

        xx=sc.transform(x)

        xx=torch.from_numpy(xx).type(torch.float32)
        
        out=model(xx)

        out=out.reshape(-1,5)

        out=torch.argmax(out, dim=1).reshape((1,-1))

        self.rule[0,0]=digi_str(out)

        for i in range(13):
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
    
    
    





    
        
    
        
            