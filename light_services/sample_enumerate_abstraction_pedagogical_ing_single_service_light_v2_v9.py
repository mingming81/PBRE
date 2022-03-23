import torch 
import numpy as np


# from tree_ing_2 import *
from tree_ing_4_single_service_light_v2_v9_p2 import *

max_x_light=1005
# %%
X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)

X_LS=torch.tensor([[0,1,2,3,4]],dtype=torch.float32,device=device)

X_CUR=torch.tensor([[0,1/2,1]],dtype=torch.float32, device=device)

# X: torch tensor 1*n
def one_hot_decoder(X):
    idx=(X == 1).nonzero(as_tuple=False)
    num=idx[0,1].reshape((1,1))
    return num

def digi_str(x_ls,x_cur,x_us):
    x_ls_str=np.full([1,2], None)
    x_cur_str=np.full([1,2], None)
    
    x_us_str=np.full([1,2], None)
    # x_lr_str=np.full([1,2], None)
    # x_le_str=np.full([1,2], None)
    
    # if x_ls[0,0]==torch.tensor(0):
    #     x_ls_str[0,0]='lamp is off'
    # elif x_ls[0,0]==torch.tensor(1):
    #     x_ls_str[0,0]='lamp is at level 1'
    # elif x_ls[0,0]==torch.tensor(2):
    #     x_ls_str[0,0]='lamp is at level 2'
    # elif x_ls[0,0]==torch.tensor(3):
    #     x_ls_str[0,0]='lamp is at level 3'
    # elif x_ls[0,0]==torch.tensor(4):
    #     x_ls_str[0,0]='lamp is at level 4'
    x_ls_str[0,0]='lamp' 
    x_ls_str[0,1]=x_ls.cpu()
    
    
    # if x_cur[0,0]==torch.tensor(0)/2:
    #     x_cur_str[0,0]='curtain is closed'
    # elif x_cur[0,0]==torch.tensor(1)/2:
    #     x_cur_str[0,0]='curtain is half open'
    # elif x_cur[0,0]==torch.tensor(2)/2:
    #     x_cur_str[0,0]='curtain is fully open'
    x_cur_str[0,0]='curtain'
    x_cur_str[0,1]=x_cur.cpu()
        
    # if x_us[0,0]==torch.tensor(0):
    #     x_us_str[0,0]='user is absent'
    # elif x_us[0,0]==torch.tensor(1):
    #     x_us_str[0,0]='user is working'
    # elif x_us[0,0]==torch.tensor(2):
    #     x_us_str[0,0]='user is entertaining'
    # elif x_us[0,0]==torch.tensor(3):
    #     x_us_str[0,0]='user is sleeping'
    
    x_us_str[0,0]='user'
        
    x_us_str[0,1]=x_us.cpu()
    
    # x_lr_str[0,0]=f'the indoor light is {x_lr.item()}'
    # x_lr_str[0,1]=x_lr.cpu()
    
    # x_le_str[0,0]=f'the outdoor light intensity'
    # x_le_str[0,1]=x_le.cpu()
    
    return x_ls_str,x_cur_str,x_us_str
    
    
# seapeda: sample enumerate abstraction pedagogical method
class Seapeda:
    def __init__(self):
        self.ruleTree=Node('root')
    
    def ruleExtract(self,x,model):
        
        
            
        self.rule=np.full([1,3],None)
        
        
        
        x=x.reshape((1,-1))
        out=model(x)
        
        X_ls=out[0,:X_LS.shape[1]].reshape((1,-1))
        x_ls=torch.argmax(X_ls).reshape((1,1))
        
        
        X_cur=out[0,X_LS.shape[1]:].reshape((1,-1))
        x_cur=(torch.argmax(X_cur)/2).reshape((1,1))

        if x_cur.item()==1.5:
            print(f'x: {x}')
            import sys
            sys.exit()

        # if True:
        #     # the user is 100% sure that when he is absent or sleeping, the lamp
        #     # should be off
            
        #     if ((x[:,:len(X_US[0])]).max(1)[1].item())==0:
        #         x_ls=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         x_cur=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         # idx_x_cur_t_new=torch.tensor([[0]],device=device)

        #     if ((x[:,:len(X_US[0])]).max(1)[1].item())==3:
        #         x_ls=torch.tensor([[0]],dtype=torch.float32,device=device)

        X_us=x[0,:X_US.shape[1]].reshape((1,-1))
        x_us=one_hot_decoder(X_us)
        
        # x_lr=x[0,4:5].reshape((1,-1))
        # x_lr=x_lr*max_x_light
        
        # x_le=x[0,4:].reshape((1,-1))
        # x_le=x_le*max_x_light
        
        x_ls_str,x_cur_str,x_us_str=digi_str(x_ls, x_cur, x_us)
        
        self.rule[0,0]=x_ls_str
        self.rule[0,1]=x_cur_str
        self.rule[0,2]=x_us_str
        # self.rule[0,3]=x_lr_str
        # self.rule[0,3]=x_le_str
        
        return self.rule
        
        
    
    
    def update(self,rule):
        
        self.ruleTree.update(toCheck=rule)
        
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
    
    
    





    
        
    
        
            