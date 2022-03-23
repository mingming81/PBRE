# Acuire rewards for each epoch, state transition caused by the rules extracted from
# pedagogical rule extraction method. (rule_generation)
# %%
import torch
import torch.nn.functional as F
# import numpy as np

from collections import deque
import random
import copy
import math
import os
import pickle

import numpy as np

from sample_enumerate_abstraction_pedagogical_ing_single_service_light_lstm import *
# from rule_generation import *

torch.autograd.set_detect_anomaly(True)

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')

# %%
# create a data storing path

dirName='11082021/model_save'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='11082021/section6_seen'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)


# if os.path.exists("./../v3/"+other_data_name+"/logicalRuleEvaluation_v2_48_with_seen_100.log"):
#     f = open("./../v3/"+other_data_name+"/logicalRuleEvaluation_v2_48_with_seen_100.log", "r+")
# else:
#     f = open("./../v3/"+other_data_name+"/logicalRuleEvaluation_v2_48_with_seen_100.log", "w")


# import logging

# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler("./../v3/"+other_data_name+"/logicalRuleEvaluation_v2_48_with_seen_100.log", 'a'))
# print = logger.info

# %%
# 1) uses the reaction of the inhabitant to replace the direct action choice given 
# the user
# 2) introudce the percentage confidence choice: for example, the user is 100% sure 
# that the lamp should be off when the user is absent. or the user is 70% sure that
# the lamp shouls be in level 3 when the user is entertaining.

# %%
# user states

X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)

# %%
# lamp state
X_LS=torch.tensor([[0,1,2,3,4]],dtype=torch.float32,device=device)

# %%
# curtain state
X_CUR=torch.tensor([[0,1/2,1]],dtype=torch.float32, device=device)



lenXus=X_US.shape[1]

# %%
# lamp state

lenXls=X_LS.shape[1]
# %%
# curtain state

lenXcur=X_CUR.shape[1]
# %%
# parameters

# train the model

batch_size=120
# 40000
num_epochs=100
steps=288

learning_rate=0.001

epsilon=0.1

max_x_light=1005

# %%
# user state

X_us_t=torch.tensor([[3,3,3,3,3,3,3,2,0,1,1,1,2,0,1,1,1,1,2,0,0,2,2,2]],dtype=torch.float32,device=device)

def X_us_t_generation():
    X_us_t=torch.randint(0,4,(1,1)).to(device)
    return X_us_t

# X_us_t=X_us_t_generation()
# X_us_t=X_us_t.reshape((-1,1))

# %%
# initial indoor light intensity
x_lr_0=torch.tensor([[0]],dtype=torch.float32, device=device)

# %%
# outside light intensity
# X_le_t=torch.tensor([[0,0,0,0,0,200,400,500,600,700,800,800,900,900,800,700,600,500,400,200,0,0,0,0]],dtype=torch.float32,device=device)

def X_le_t_generation(max_x_light2=600):


    m = modeling.models.Gaussian1D(amplitude=max_x_light2, mean=12, stddev=3)
    x = np.linspace(0, 24, steps)
    data = m(x)
    data = data + 5*np.random.random(x.size)
    data=data.astype(np.float32)
    
    x_le_t=torch.from_numpy(data).reshape((1,-1))

    # print(f'x_le_t: {x_le_t.shape}')
    return x_le_t
    
# X_le_t=X_le_t_generation()
# X_le_t=X_le_t.reshape((-1,1))

# %%
# lamp stte


# %%
# deta_t=5 (mins)=5/60 (h): at each time step, each actuator work for five 
# minute, and hour is preferred in this example.

# setStates(): get related states related with the light intensity (
#              before updating, actuator states not change yet)
# getActions(): get model predictions (actuator states change)

# updateStates(): update indoor light intensities with new other actuator states



class LightService:
    
    def __init__(self, beta=100, replayMemorySize=1000000,
                 minibatchSize=128,discount=0.1,learningRate=0.9):
        self.beta=beta
        self.replayMemorySize=replayMemorySize
        self.minibatchSize=minibatchSize
        self.discount=discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        self.replayMemory=deque(maxlen=self.replayMemorySize)
        
        self.lightServiceModel=self.createModel().to(device)
        
        
        self.optimizer=torch.optim.Adam(self.lightServiceModel.parameters(),lr=0.1)
        
        self.targetLightServiceModel=self.createModel().to(device)
        self.targetLightServiceModel.load_state_dict(self.lightServiceModel.state_dict())
        
        
        self.targetLightServiceModel.eval()
        
        
        # %%

    def setRules(self,logicalRules):
        self.logicalRules=logicalRules
        
    def updateReplayMemory(self,transition):
        self.replayMemory.append(transition)
        
        # %%
        
    def setStates(self,x_us_t,x_lr_t,x_le_t,x_ls_t,x_cur_t,MAX_LE):
        self.x_us_t=x_us_t
        self.x_lr_t=x_lr_t
        self.x_le_t=x_le_t
        self.x_ls_t=x_ls_t
        self.x_cur_t=x_cur_t
        
        self.MAX_LE=MAX_LE
        
        # %%
        
    def getRewards(self):
        
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            if self.x_ls_t==torch.tensor([[0]],dtype=torch.float32,device=device) and self.x_cur_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=4
            else:
                self.reward=-4
                
        if self.x_us_t==torch.tensor([[1]],dtype=torch.float32,device=device):
            if self.x_lr_t>=torch.tensor([[300-50]],dtype=torch.float32,device=device) and self.x_lr_t<=torch.tensor([[300+50]],dtype=torch.float32,device=device):    
                if self.x_ls_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=8
                
                else:
                    
                    x_ls_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    for i in range(X_CUR.shape[1]):
                        if stop:
                            break
                        x_cur_new_t_int=X_CUR[[[0]],i]
                        x_lr_new_t_int=self.getIndoorLight(x_ls_new_t_int,x_cur_new_t_int)
                        if x_lr_new_t_int>=torch.tensor([[300-50]],dtype=torch.float32,device=device) and x_lr_new_t_int<=torch.tensor([[300+50]],dtype=torch.float32,device=device):
                            self.reward=-8
                            stop=True
                            break
                        else:
                            self.reward=4
            else:
                self.reward=-4
                
        if self.x_us_t==torch.tensor([[2]],dtype=torch.float32,device=device):
            if self.x_lr_t>=torch.tensor([[400-50]],dtype=torch.float32,device=device) and self.x_lr_t<=torch.tensor([[400+50]],dtype=torch.float32,device=device):
                if self.x_ls_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=8
                else:
                    x_ls_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    for i in range(X_CUR.shape[1]):
                        if stop:
                            break
                        x_cur_new_t_int=X_CUR[[[0]],i]
                        x_lr_new_t_int=self.getIndoorLight(x_ls_new_t_int,x_cur_new_t_int)
                        if x_lr_new_t_int>=torch.tensor([[400-50]],dtype=torch.float32,device=device) and x_lr_new_t_int<=torch.tensor([[400+50]],dtype=torch.float32,device=device):
                            self.reward=-8
                            stop=True
                            break
                        else:
                            self.reward=4
            else:
                self.reward=-4
        
        if self.x_us_t==torch.tensor([[3]],dtype=torch.float32,device=device): 
            if self.x_ls_t==torch.tensor([[0]],dtype=torch.float32,device=device) and self.x_cur_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=4
            else:
                self.reward=-4
                
        return self.reward
    
    # %%
                              
    def getIndoorLight(self,x_ls_new_t,x_cur_new_t):
        
        x_lr_new_t=torch.minimum(self.beta*x_ls_new_t+x_cur_new_t*self.x_le_t,torch.tensor([[self.MAX_LE]]))
        
        return x_lr_new_t
    
    # %%
        
    def createModel(self):
        
       lightServiceModel=LstmServiceModel()
        
       return lightServiceModel
   
   # %%
    
    def getActions(self,X_t_norm):
        
        Q=self.lightServiceModel(X_t_norm.float())
        Q_cur_t=Q[:,:len(X_CUR[0])]
        Q_ls_t=Q[:,len(X_CUR[0]):]
        
        Q_ls_t=Q_ls_t.reshape(1,-1)
        Q_cur_t=Q_cur_t.reshape(1,-1)
        
        return Q_cur_t,Q_ls_t

    def getActionsRule(self,x_light_t_norm):
        
        x_ls_new_t=np.full([1,1],None)
        x_cur_new_t=np.full([1,1],None)
        
        
        # x_us_t_norm=x_light_t_norm[0,:lenXus]
        # x_us_t_norm=x_us_t_norm.reshape(1,lenXus)
        # x_us_t=one_hot_decoder(x_us_t_norm)
        
        
        # x_lr_t_norm=x_light_t_norm[0,lenXus:lenXus+1].reshape((1,1))
        # x_lr_t=x_lr_t_norm*max_x_light
        
        # x_le_t_norm=x_light_t_norm[0,lenXus:].reshape((1,1))
        # x_le_t=x_le_t_norm*max_x_light
        
        lenRules=self.logicalRules.shape[0]
        # print('=============================')
        # print(lenRules)
        # print('=============================')





        idx_toDecide=np.full([1,1],None)
        num_rules=0

        # print(self.logicalRules[0,0][0,0].shape)
        
        for i in range(lenRules):

            if self.logicalRules[i,2][0,0][0,1]==self.x_us_t[0,0]:
                # print('one')
                if (self.x_le_t[0,0]>=self.logicalRules[i,3][0,0][0,1] and self.x_le_t[0,0]<=self.logicalRules[i,3][0,0][0,2]):

                    
                
                    num_rules+=1

                    if (idx_toDecide.shape==(1,1)):
                        idx_toDecide=copy.deepcopy(self.logicalRules[[i],:])
                        # print('idx_toDecide shape: ',idx_toDecide.shape)

                    else:
                        
                        idx_toDecideInt=np.full([1,1],None)
                        idx_toDecideInt=copy.deepcopy(self.logicalRules[[i],:])
                        idx_toDecide=np.concatenate((idx_toDecide,idx_toDecideInt),axis=0)

        # if x_us_t[0,0]==1 and math.floor(x_le_t[0,0].item())==92 :
        #     print(idx_toDecide)
        #     import sys
        #     sys.exit()


        len_idx_toDecide=idx_toDecide.shape[0]
            
        # print(f'================{num_rules}==================')

        if num_rules==1:

            x_ls_new_t=idx_toDecide[0,0][0,0][0,1]
            # print(self.logicalRules[i][0])
            # import sys
            # sys.exit()
            
            x_cur_new_t=idx_toDecide[0,1][0,0][0,1]

        
        elif num_rules>1:

            print('enter')

            maxIdx=np.full([1,1],None)
            maxIdx[0,0]=0

            idx_toDecideInt=sorted(idx_toDecideInt,key=lambda x: x[0][0,1]+x[1][0,1],reverse=True)
    
            idx_toDecideInt=np.array(idx_toDecideInt).reshape(-1,4)

            x_ls_new_t=idx_toDecideInt[0,0][0,0][0,1]
            x_cur_new_t=idx_toDecideInt[0,1][0,0][0,1]

            print(x_ls_new_t,x_cur_new_t)

            # print(x)  
            # import sys
            # sys.exit()
        
        return x_cur_new_t,x_ls_new_t
    
        
      # %%
      
    def train(self,epoch):
        
        if len(self.replayMemory)<self.minibatchSize:
            return
        
        minibatch=random.sample(self.replayMemory,self.minibatchSize)
        
        states_list_tt=[transition[0] for transition in minibatch]
        
        states_list_t=None
        
        for i in range(len(states_list_tt)):
            if i==0:
                states_list_t=states_list_tt[i]
            else:
                states_list_ttt=states_list_tt[i]
                states_list_t=torch.cat((states_list_t,states_list_ttt),axis=0)
                
        states_list_t=states_list_t.reshape((self.minibatchSize,-1))
        
        q_list_t=self.lightServiceModel(states_list_t.float()).reshape((self.minibatchSize,-1))
        q_cur_list_t=q_list_t[:,:len(X_CUR[0])]
        q_ls_list_t=q_list_t[:,len(X_CUR[0]):]
        
        
        states_list_tt_plus_1=[transition[3] for transition in minibatch]
        states_list_t_plus_1=None
        
        for i in range(len(states_list_tt_plus_1)):
            if i==0:
                states_list_t_plus_1=states_list_tt_plus_1[i]
            
            else:
                states_list_ttt_plus_1=states_list_tt_plus_1[i]
                states_list_t_plus_1=torch.cat((states_list_t_plus_1,states_list_ttt_plus_1),axis=0)
        
        states_list_t_plus_1=states_list_t_plus_1.reshape((self.minibatchSize,-1))
        
        q_list_t_plus_1=self.targetLightServiceModel(states_list_t_plus_1.float())[:,:].detach().reshape((self.minibatchSize,-1))
        
        q_cur_list_t_plus_1=q_list_t_plus_1[:,:len(X_CUR[0])]
        q_ls_list_t_plus_1=q_list_t_plus_1[:,len(X_CUR[0]):]
        
        
        X=None
        Y=None
        
        for index, (states_t,actions,reward_t,states_t_plus_1) in enumerate(minibatch):
            
            max_q_ls_t_plus_1=torch.max(q_ls_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_ls_t_plus_1=reward_t+self.discount*max_q_ls_t_plus_1
            
            max_q_cur_t_plus_1=torch.max(q_cur_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_cur_t_plus_1=reward_t+self.discount*max_q_cur_t_plus_1
            
            q_ls_t=q_ls_list_t[index,:]
            q_ls_t=q_ls_t.reshape(1,-1)
            
            q_cur_t=q_cur_list_t[index,:]
            q_cur_t=q_cur_t.reshape(1,-1)
            
            action_cur_t,action_ls_t=actions
            
            action_ls_t_item=action_ls_t.item()
            action_ls_t_item=(X_LS==action_ls_t_item).nonzero(as_tuple=True)[1].item()
            
            action_cur_t_item=action_cur_t.item()
            action_cur_t_item=(X_CUR==action_cur_t_item).nonzero(as_tuple=True)[1].item()
            
            q_ls_t[0,int(action_ls_t_item)]=(1-self.learningRate)*q_ls_t[0,int(action_ls_t_item)]+self.learningRate*new_q_ls_t_plus_1
            
            # q_ls_t[0,:int(action_ls_t_item)]=self.learningRate*q_ls_t[0,:int(action_ls_t_item)]-(1-self.learningRate)*reward_t
            # q_ls_t[0,int(action_ls_t_item)+1:]=self.learningRate*q_ls_t[0,int(action_ls_t_item)+1:]-(1-self.learningRate)*reward_t
            
            q_cur_t[0,int(action_cur_t_item)]=(1-self.learningRate)*q_cur_t[0,int(action_cur_t_item)]+self.learningRate*new_q_cur_t_plus_1
            # q_cur_t[0,:int(action_cur_t_item)]=self.learningRate*q_cur_t[0,:int(action_cur_t_item)]-(1-self.learningRate)*reward_t
            # q_cur_t[0,int(action_cur_t_item)+1:]=self.learningRate*q_cur_t[0,int(action_cur_t_item)+1:]-(1-self.learningRate)*reward_t
            
            
            
            if index==0:
                X=copy.deepcopy(states_t)
                Y=torch.cat((q_cur_t,q_ls_t),axis=1)
            
            else:
                X=torch.cat((X,states_t),axis=0)
                q_t=torch.cat((q_cur_t,q_ls_t),axis=1)
                Y=torch.cat((Y,q_t),axis=0)
                
        data_size=len(X)
        validation_pct=0.2
        
        train_size=math.ceil(data_size*(1-validation_pct))
        
        X_train,Y_train=X[:train_size,:],Y[:train_size,:]
        X_test,Y_test=X[train_size:,:],Y[train_size:,:]
        
        if epoch>=0:
            self.optimizer=torch.optim.Adam(self.lightServiceModel.parameters())
            
        least_val_loss=0
        total_patience=50
        patience=0
        
        outputs=self.lightServiceModel(X_train.float()).reshape((train_size,-1))
        
        criterion=torch.nn.MSELoss()
        
        loss=criterion(outputs,Y_train)
        
        self.optimizer.zero_grad()
        
        loss.backward(retain_graph=True)
        
        self.optimizer.step()
        
        criterion_val=torch.nn.MSELoss()
        
        
        self.lightServiceModel.eval()
        
        total_val_loss=torch.tensor([[0]],dtype=torch.float32,device=device)
        
        with torch.no_grad():
            n_correct=0
            n_samples=0
            # for index in range(len(X_test)):
            x_test=X_test
            y_test=Y_test
            
            outputs=self.lightServiceModel(x_test.float()).reshape((x_test.shape[0],-1))

            # print(f'x_test shape: {x_test.shape}')
            # import sys
            # sys.exit()
            # outputs=torch.cat((outputs_ls,outputs_cur),axis=1)
            val_loss=criterion_val(outputs,y_test)
            # val_loss=criterion_val(outputs,y_test)
            # total_val_loss=total_val_loss+val_loss
                
            print(f'val loss: {val_loss.item():.4f}, train loss: {(loss.item()):.4f}')
            
            if epoch==0:
                least_val_loss=total_val_loss
            else:
                if least_val_loss>total_val_loss:
                    least_val_loss=total_val_loss
                else:
                    patience+=1
            
            if patience==50:
                torch.save(self.lightServiceModel.state_dict(),dirName+f'/light_model_structure_v2.pth')
                print("end training")
                return
            
        # print('==========================test end=====================')
        
        self.lightServiceModel.train()
        
        return minibatch

   
 
# %%
class LstmServiceModel(torch.nn.Module):
    def __init__(self,hidden_layers=20):
        super(LstmServiceModel,self).__init__()
        
        self.num_output_features=8
        
        self.hidden_layers=hidden_layers
        
        self.lstm1=torch.nn.LSTMCell(5,self.hidden_layers)
        self.lstm2=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.lstm3=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.linear=torch.nn.Linear(self.hidden_layers,self.num_output_features)
            
    def forward(self,y):
       
        
        if len(y.shape)!=3:
            y=torch.from_numpy(np.expand_dims(y, axis=0))
            
    
        n_samples =y.size(0)
        
        h_t, c_t = self.lstm1(y[0])
        h_t2, c_t2 = self.lstm2(h_t)
        h_t3, c_t3 = self.lstm2(h_t2)
        
        output = self.linear(h_t3)
       

        return output
    
    

    


# %%

class OneHotEncoderClass:
    def __init__(self):
        pass
    def _one_hot_encoder(self,X,x):
        
        zeros=torch.zeros(X.shape,dtype=torch.float32,device=device)
        # print(f'zeros shape: {zeros.shape}')
        pos=torch.where(X==x)[1].item()
        zeros[0,pos]=1
        one_hot_encod=zeros
        
        return one_hot_encod
    
    
# %%
# define the service
LightService=LightService()
LightService.lightServiceModel.load_state_dict(torch.load('./../v3/'+dirName+f'/lightService_light_lstm_v2_sig.pth'))
LightService.targetLightServiceModel.load_state_dict(torch.load('./../v3/'+dirName+f'/lightService_light_lstm_v2_sig.pth'))


logicalRules=np.load('./../v3/'+other_data_name+'/logical_rules_light_service_structure_lstm.npy',allow_pickle=True)
print(f"extracted rule shape: {logicalRules}")
print(f"extracted rule shape: {logicalRules.shape}")

import sys
sys.exit()

LightService.setRules(logicalRules)

replayMemory=np.load('./../v3/'+other_data_name+'/replayMemory_light_lstm_unseen.npy',allow_pickle=True)
# print(f'replayMemory shape: {replayMemory.shape}')

currentStates=[transition[0] for transition in replayMemory]
MAX_LEs=[transition[-1] for transition in replayMemory]

numericalEpochReward=np.load('./../v3/'+other_data_name+'/totalEpochRewards_light_lstm_unseen.npy',allow_pickle=True)
# print(f'totalEpochReward shape: {numericalEpochReward.shape}')
# %%

totalEpochRewards=np.full([1,num_epochs],0)
notSeenStates=np.full([1,1],None)

# %%

for epoch in range(num_epochs):
    print(f'epoch: {epoch}')

    OneHotEncoder=OneHotEncoderClass()
    
    x_ls_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_cur_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_lr_t=x_lr_0
    
    for i in range(0,steps):
        
        step=epoch*steps+i
        
        if step==74:
            print('yes...')
            
        print(f'step: {step}')
        
        x_light_t_norm=currentStates[step].to(device)

        MAX_LE=MAX_LEs[step]
        
  
        x_us_t_norm=x_light_t_norm[0,:lenXus]
        x_us_t_norm=x_us_t_norm.reshape(1,-1)
        # print(f'x_light_t_norm: {x_light_t_norm}')
        x_us_t=one_hot_decoder(x_us_t_norm)
        
        
        # x_lr_t_norm=x_light_t_norm[0,lenXus:lenXus+1].reshape((1,-1))
        # x_lr_t=x_lr_t_norm*max_x_light
        
        # x_le_t_norm=x_light_t_norm[0,lenXus:].reshape((1,-1))
        x_le_t=((x_light_t_norm[0,lenXus:]*MAX_LEs[step]).reshape((1,-1))).to(device)

        # print(x_us_t,x_le_t)
        
        
        # if step==78:
        #     print(x_us_t,x_le_t)

        # if step==80:
        #     import sys
        #     sys.exit()
       
        x_ls_t_new=np.full([1,1],None)
        x_cur_t_new=np.full([1,1],None)

        LightService.setStates(x_us_t,x_lr_t,x_le_t,x_ls_t,x_cur_t, MAX_LE)
        
        
        # difference
        x_cur_t_new,x_ls_t_new=LightService.getActionsRule(x_light_t_norm)

      
        # if True:
        
        #     if ((x_light_t_norm[:,:len(X_US[0])]).max(1)[1].item())==0:
        #         x_ls_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         x_cur_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         # idx_x_cur_t_new=torch.tensor([[0]],device=device)

        #     if ((x_light_t_norm[:,:len(X_US[0])]).max(1)[1].item())==3:
        #         x_ls_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)

        
        if x_ls_t_new[0,0]==None or x_cur_t_new[0,0]==None:
            print("no rules ==================================")
            
            # print('============================================')
            if notSeenStates[0,0]==None:
                notSeenStates[0,0]=step
                notSeenStates=notSeenStates.astype(np.int)
            else:
                notSeenStatesInt=np.array([[step]])
                notSeenStates=np.concatenate((notSeenStates,notSeenStatesInt),axis=1)
            
            continue
    
        # if True:

        #     if ((x_light_t_norm[:,:len(X_US[0])]).max(1)[1].item())==0:
        #         x_ls_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         x_cur_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)

        #     if ((x_light_t_norm[:,:len(X_US[0])]).max(1)[1].item())==3:
        #         x_ls_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
        
        x_lr_t_new=LightService.getIndoorLight(x_ls_t_new,x_cur_t_new)
        
        x_us_t_new=x_us_t
        x_lr_t_new=x_lr_t_new
        x_le_t_new=x_le_t
        x_ls_t_new=x_ls_t_new
        x_cur_t_new=x_cur_t_new
        
        x_us_t_new_norm=OneHotEncoder._one_hot_encoder(X_US,x_us_t_new)
        x_lr_t_new_norm=x_lr_t_new/MAX_LE
        x_le_t_new_norm=x_le_t_new/MAX_LE
        x_light_t_new_norm=torch.cat((x_us_t_new_norm,
                             # x_lr_t_new_norm,
                             x_le_t_new_norm),axis=1)

        # x_light_t_new_norm=copy.deepcopy(x_us_t_new_norm)
        
        actions=(x_cur_t_new, x_ls_t_new)
        
        LightService.setStates(x_us_t_new,x_lr_t_new,x_le_t_new,x_ls_t_new,x_cur_t_new,MAX_LE)
        reward=LightService.getRewards()
        print(f'reward: {reward}')
        
        totalEpochRewards[0,epoch]+=reward
        
        reward_t=torch.tensor([[reward]],dtype=torch.float32,device=device)
        
        transition=(x_light_t_norm,actions,reward_t,x_light_t_new_norm,x_le_t)
        LightService.updateReplayMemory(transition)
        
     
        # LightService.train(epoch,step)
        
        
        x_lr_t=x_lr_t_new
        x_ls_t=x_ls_t_new
        x_cur_t=x_cur_t_new
        
    # if epoch%10==0:
    #     LightService.targetLightServiceModel.load_state_dict(LightService.LightServiceModel.state_dict())

# %%
# epoch reward saving 
# np.save('./../v3/section6/not_seen_states'+'/totalEpochRewards_2_v3_with_not_seen_RULES_100.npy',totalEpochRewards)

# np.save('./../v3/section6/not_seen_states'+'/replayMemory_2_v3_with_not_seen_RULES_100.npy',LightService.replayMemory)

# np.save('./../v3/section6/not_seen_states'+'/states_with_not_seen_RULES_100.npy',notSeenStates)

print('=====================================totalEpochRewards=======================')

print(f'totalEpochRewards: {totalEpochRewards}')

print('=====================================notSeenStates=======================')
# 10 epochs
print(notSeenStates.shape)
print(notSeenStates)


# np.save('./../v3/'+other_data_name+'/totalEpochRewards_light_service_structure_lstm_RULES_100_seen.npy',totalEpochRewards)

# np.save('./../v3/'+other_data_name+'/replayMemory_light_service_structure_lstm_RULES_100_seen.npy',LightService.replayMemory)

# np.save('./../v3/'+other_data_name+'/notseen_states_light_service_structure_lstm_RULES_100_seen.npy',notSeenStates)


np.save('./../v3/'+other_data_name+'/totalEpochRewards_light_service_structure_lstm_RULES_100_unseen.npy',totalEpochRewards)

np.save('./../v3/'+other_data_name+'/replayMemory_light_service_structure_lstm_RULES_100_unseen.npy',LightService.replayMemory)

np.save('./../v3/'+other_data_name+'/notseen_states_light_service_structure_lstm_RULES_100_unseen.npy',notSeenStates)

