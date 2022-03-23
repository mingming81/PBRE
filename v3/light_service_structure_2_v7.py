# this version adds one more hidden layer in the neural networks, and adds the condition that the curtain should be closed when the user is sleeping


import torch
import torch.nn.functional as F
# import numpy as np

from collections import deque
import random
import copy
import math
import os
from astropy import modeling

import numpy as np

random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%
# create a data storing path


dirName='./11082021/model_save'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='./11082021/section6_seen'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)

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

# %%
# parameters

# train the model

batch_size=120
# 40000
num_epochs=10000
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

    # if step<int(5*60/5):
    #     X_le_t=random.randrange(10)
    # if step>=int(5*60/5) and step<(7*60/5):
    #     X_le_t=random.randrange(30,100)
    # if step>=int(7*60/5) and step<int(9*60/5):
    #     x_le_t=random.randrange(90,200)
    # if step>=int(9*60/5) and step<int(18*60/5):
    #     X_le_t=random.randrange(250,600)
    # if step>=int(18*60/5) and step<int(21*60/5):
    #     X_le_t=random.randrange(100,300)
    # if step>=int(21*60/5) and step<int(23*60/5):
    #     X_le_t=random.randrange(100,150)
    # if step>=int(23*60/5) and step<int(24*60/5):
    #     X_le_t=random.randrange(0,30)
    # X_le_t=torch.tensor([[X_le_t]])
    # return X_le_t
    
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
    
    def __init__(self,beta=100,power=60,deta_t=5/60,ReplayMemorySize=1000000,
                 MinibatchSize=128,Discount=0.1, learningRate=0.9):
        
        self.beta=beta
        self.power=power
        self.delta_t=deta_t
        self.MinibatchSize=MinibatchSize
        self.ReplayMemorySize=ReplayMemorySize
        self.Discount=Discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        
        self.replayMemory=deque(maxlen=self.ReplayMemorySize)
        
        self.LightServiceModel=self.createModel().to(device)
        # self.LightServiceModel.load_state_dict(torch.load(dirName+f'/model_structure_2_v3_28.pth'))
        # for p in self.LightServiceModel.parameters():
        #     p.requires_grad=True
        
        # before the target model definitions
        self.optimizer=torch.optim.Adam(self.LightServiceModel.parameters(),lr=0.1)
        


        
        self.targetLightServiceModel=self.createModel().to(device)
        # self.targetLightServiceModel.load_state_dict(torch.load(dirName+f'/model_structure_2_v3_28.pth'))

        self.targetLightServiceModel.load_state_dict(self.LightServiceModel.state_dict())
        

        
        self.targetLightServiceModel.eval()
        

        # for p in self.targetLightServiceModel.parameters():
        #     p.requires_grad=False
        
        
        
        
    def updateReplayMemory(self,transition):
        self.replayMemory.append(transition)
        
    def setStates(self,x_us_t,x_lr_t,x_le_t,x_ls_t,x_cur_t):
        self.x_us_t=x_us_t
        self.x_lr_t=x_lr_t
        self.x_le_t=x_le_t
        self.x_ls_t=x_ls_t
        self.x_cur_t=x_cur_t
        
        
    def getStates(self):
        
        return self.x_us_t,self.x_lr_t,self.x_le_t,self.x_ls_t,self.x_cur_t
        
    
    def getReward(self,X_t_norm):
        
        # x_ls_new_t,x_cur_new_t=self.getActions(X_t_norm)
        # x_lr_new_t=self.getIndoorLight(x_ls_new_t,x_cur_new_t)
        # self.setStates(self.x_us_t, x_lr_new_t, self.x_le_t, x_ls_new_t, x_cur_new_t)
        
        # ===================================================================================
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            if self.x_ls_t==torch.tensor([[0]],dtype=torch.float32,device=device) and self.x_cur_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=4
            else:
                self.reward=-4
        # ===================================================================================
        if self.x_us_t==torch.tensor([[1]],dtype=torch.float32,device=device):
            if self.x_lr_t>=torch.tensor([[300-50]],dtype=torch.float32,device=device) and self.x_lr_t<=torch.tensor([[300+50]],dtype=torch.float32,device=device):
                x_ls_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                for i in range(X_CUR.shape[1]):
                    x_cur_new_t_int=X_CUR[[[0]],i]
                    x_lr_new_t_int=self.getIndoorLight(x_ls_new_t_int,x_cur_new_t_int)
                    if x_lr_new_t_int>=torch.tensor([[300-50]],dtype=torch.float32,device=device) and x_lr_new_t_int<=torch.tensor([[300+50]],dtype=torch.float32,device=device):
                        # print("caseeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                        self.reward=2
                    else:
                        self.reward=4
            else:
                self.reward=-4
                
        # ===================================================================================
        if self.x_us_t==torch.tensor([[2]],dtype=torch.float32,device=device):
            if self.x_lr_t>=torch.tensor([[400-50]],dtype=torch.float32,device=device) and self.x_lr_t<=torch.tensor([[400+50]],dtype=torch.float32,device=device):
                x_ls_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                for i in range(X_CUR.shape[1]):
                    x_cur_new_t_int=X_CUR[[[0]],i]
                    x_lr_new_t_int=self.getIndoorLight(x_ls_new_t_int,x_cur_new_t_int)
                    if x_lr_new_t_int>=torch.tensor([[400-50]],dtype=torch.float32,device=device) and x_lr_new_t_int<=torch.tensor([[400+50]],dtype=torch.float32,device=device):
                        # print("caseeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                        self.reward=2
                    else:
                        self.reward=4
            else:
                self.reward=-4
                
        # ===================================================================================
        if self.x_us_t==torch.tensor([[3]],dtype=torch.float32,device=device): 
            if self.x_ls_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=4
            else:
                self.reward=-4
                
        return self.reward
    
    
    def getIndoorLight(self,x_ls_new_t,x_cur_new_t):
        
        # self.x_ls_new_t,self.x_cur_new_t=self.getActions()
        
        x_lr_new_t=torch.minimum(self.beta*x_ls_new_t+x_cur_new_t*self.x_le_t,torch.tensor([[max_x_light]]))
        
        return x_lr_new_t
        
            
    def createModel(self):
        
        self.LightServiceModel=ServiceModel()
        return self.LightServiceModel
    
    def getActions(self,X_t_norm):

        # self.x_us_t,self.x_lr_t,self.x_le_t,_,_=self.getStates()
        
        # X_t=torch.cat((self.x_us_t,self.x_lr_t,self.x_le_t),axis=1)
        
        Q=self.LightServiceModel(X_t_norm)
        Q_ls_t=Q[:,:len(X_LS[0])]
        Q_cur_t=Q[:,len(X_LS[0]):]
        Q_ls_t=Q_ls_t.reshape(1,-1)
        Q_cur_t=Q_cur_t.reshape(1,-1)
        
        x_ls_new_t=torch.max(Q_ls_t,dim=1,keepdim=True)[1].to(device)
        idx_x_cur_new_t=(torch.max(Q_cur_t,dim=1,keepdim=True)[1]).to(device)
        x_cur_new_t=X_CUR[[[0]],idx_x_cur_new_t.item()]
      
        
        return x_ls_new_t,x_cur_new_t,idx_x_cur_new_t
    
    def train(self):
        
        if len(self.replayMemory) < self.MinibatchSize:
            return 
        
        minibatch=random.sample(self.replayMemory,self.MinibatchSize)
        
        states_list_tt=[transition[0] for transition in minibatch]
        
        states_list_t=None
        
        for i in range(len(states_list_tt)):
            if i==0:
                states_list_t=states_list_tt[i]
            else:
                states_list_ttt=states_list_tt[i]
                states_list_t=torch.cat((states_list_t,states_list_ttt),axis=0)
        
        states_list_t=states_list_t.reshape((self.MinibatchSize,-1))
        
        q_list_t=self.LightServiceModel(states_list_t).reshape((self.MinibatchSize,-1))
        q_ls_list_t=q_list_t[:,:len(X_LS[0])]
        q_cur_list_t=q_list_t[:,len(X_LS[0]):]
        
        states_list_tt_plus_1=[transition[3] for transition in minibatch]
        states_list_t_plus_1=None
        for i in range(len(states_list_tt_plus_1)):
            if i==0:
                states_list_t_plus_1=states_list_tt_plus_1[i]
            else:
                states_list_ttt_plus_1=states_list_tt_plus_1[i]
                states_list_t_plus_1=torch.cat((states_list_t_plus_1,states_list_ttt_plus_1),axis=0)
                
        states_list_t_plus_1=states_list_t_plus_1.reshape((self.MinibatchSize,-1))
        
        
        
        q_list_t_plus_1=self.targetLightServiceModel(states_list_t_plus_1)[:,:].detach().reshape((self.MinibatchSize,-1))
        q_ls_list_t_plus_1=q_list_t_plus_1[:,:len(X_LS[0])]
        q_cur_list_t_plus_1=q_list_t_plus_1[:,len(X_LS[0]):]
        
        X=None
        Y=None
        
        for index, (states_t,actions,reward_t,states_t_plus_1) in enumerate(minibatch):
            # if index!=23:
            max_q_ls_t_plus_1=torch.max(q_ls_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_ls_t_plus_1=reward_t+self.Discount*max_q_ls_t_plus_1
            
            max_q_cur_t_plus_1=torch.max(q_cur_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_cur_t_plus_1=reward_t+self.Discount*max_q_cur_t_plus_1
                
            # else:
            #     new_q_ls_t_plus_1=reward_t
            #     new_q_cur_t_plus_1=reward_t
            
            q_ls_t=q_ls_list_t[index,:]
            q_ls_t=q_ls_t.reshape(1,-1)
            
            q_cur_t=q_cur_list_t[index,:]
            q_cur_t=q_cur_t.reshape(1,-1)
            
            action_ls_t, action_cur_t=actions
            
            action_ls_t_item=action_ls_t.item()
            action_ls_t_item=(X_LS==action_ls_t_item).nonzero(as_tuple=True)[1].item()

            action_cur_t_item=action_cur_t.item()
            action_cur_t_item=(X_CUR==action_cur_t_item).nonzero(as_tuple=True)[1].item()
            
            q_ls_t[0,int(action_ls_t_item)]=(1-self.learningRate)*q_ls_t[0,int(action_ls_t_item)]+self.learningRate*new_q_ls_t_plus_1
            q_cur_t[0,int(action_cur_t_item)]=(1-self.learningRate)*q_cur_t[0,int(action_cur_t_item)]+self.learningRate*new_q_cur_t_plus_1
            
            if index==0:
                X=copy.deepcopy(states_t)
                Y=torch.cat((q_ls_t,q_cur_t),axis=1)
            
            else:
                X=torch.cat((X,states_t),axis=0)
                q_t=torch.cat((q_ls_t,q_cur_t),axis=1)
                Y=torch.cat((Y,q_t),axis=0)
                
        data_size=len(X)
        validation_pct=0.2
        
        train_size=math.ceil(data_size*(1-validation_pct))
        
        X_train,Y_train=X[:train_size,:],Y[:train_size,:]
        X_test,Y_test=X[train_size:,:],Y[train_size:,:]

        if epoch>=0:
            self.optimizer=torch.optim.Adam(self.LightServiceModel.parameters())
        
        # trainingSet=((X_train,Y_train))
        # testSet=(X_test,Y_test)
        
        
        
        least_val_loss=0
        total_patience=50
        patience=0
        
        # for i in range(self.num_epochs):
            
        outputs=self.LightServiceModel(X_train).reshape((train_size,-1))
        
        criterion=torch.nn.MSELoss()
        
        # outputs=torch.cat((outputs_ls,outputs_cur),axis=1)
        
        loss=criterion(outputs,Y_train)
        
        # loss.requres_grad = True
        # backward pass
        self.optimizer.zero_grad()
        
        
        
        # back gradient
        
        loss.backward(retain_graph=True)
        
        # for param in self.LightServiceModel.parameters():
        #     param.grad.data.clamp_(-1, 1)
            
        
        # weights updating
        self.optimizer.step()
        
        
        
        
        
        # print(f'epoch={epoch}, step={step}, training loss={loss.item():.4f}')
        
        # print('==========================training end=====================')
                
        # test (evaluation)
        # for this process, we do not want to compute the gradient:
        criterion_val=torch.nn.MSELoss()
        
        
        self.LightServiceModel.eval()
        
        total_val_loss=torch.tensor([[0]],dtype=torch.float32,device=device)
        with torch.no_grad():
            n_correct=0
            n_samples=0
            # for index in range(len(X_test)):
            x_test=X_test
            y_test=Y_test
            
            outputs=self.LightServiceModel(x_test).reshape((x_test.shape[0],-1))

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
            
            # if patience==50:
            #     torch.save(self.LightServiceModel.state_dict(),dirName+f'/single_service_light_v1.pth')
            #     print("end training")
            #     return
            
        # print('==========================test end=====================')
        
        self.LightServiceModel.train()
        return minibatch
   
 
# %%

class ServiceModel(torch.nn.Module):
    def __init__(self):
        super(ServiceModel,self).__init__()
        
        num_input_features=5

        num_hidden=15
       
        num_output_features=8

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(num_input_features,num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden,num_output_features),

            )

        # self.hidden=torch.nn.functional.relu(num_input_features,num_hidden)
        
        
        # self.output=torch.nn.Linear(num_hidden,num_output_features)
        
        
    def forward(self,x):
        x = self.flatten(x)
        y=self.linear_relu_stack(x)

        return y
    


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

# try:

# define the service
LightService=LightService()

# LightService.LightServiceModel.load_state_dict(torch.load(dirName+f'/single_service_light_v1_32_v3.pth'))
# LightService.targetLightServiceModel.load_state_dict(torch.load(dirName+f'/single_service_light_v1_32_v3.pth'))

seenTrainingStates=deque(maxlen=100000)


# %%
# training process
totalEpochRewards=np.full([1,num_epochs],0)

for epoch in range(num_epochs):

    # print(f'epoch: {epoch}')
    
    # initialize the environment

    OneHotEncoder=OneHotEncoderClass()
    
    x_ls_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_cur_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_lr_t=x_lr_0
    X_le_t=X_le_t_generation()
    
    
    for step in range(0,steps):
        
        if step==0:
        
            x_us_t=X_us_t[0,step]
            x_us_t=x_us_t.reshape(1,-1)
            
            x_lr_t=x_lr_t
            
            x_le_t=X_le_t[0,step]
            x_le_t=x_le_t.reshape((1,-1))
            
            x_ls_t=x_ls_t
            x_cur_t=x_cur_t
            
        else:
            
            if step%12==0:
                x_us_t=X_us_t_generation()
            x_us_t=x_us_t.reshape((1,-1))
            
            x_lr_t=x_lr_t
            
            x_le_t=X_le_t[0,step]
            x_le_t=x_le_t.reshape((1,-1))
            
            x_ls_t=x_ls_t
            x_cur_t=x_cur_t
            
            
            

        
        LightService.setStates(x_us_t,x_lr_t,x_le_t,x_ls_t,x_cur_t)
        
        x_us_t_norm=OneHotEncoder._one_hot_encoder(X_US,x_us_t)
        
        
    
        x_lr_t_norm=x_lr_t/max_x_light
        x_le_t_norm=x_le_t/max_x_light
        x_light_t_norm=torch.cat((x_us_t_norm,
                            
                             x_le_t_norm),axis=1)
        
        sigma=torch.rand(1).item()
        
        # from general to specific
          
        if sigma>0.1:
            
            x_ls_t_new,x_cur_t_new,idx_x_cur_t_new=LightService.getActions(x_light_t_norm)
            
        if sigma<0.1:
            x_ls_t_new=torch.randint(0,len(X_LS[0]),(1,1)).to(device)
            idx_x_cur_t_new=(torch.randint(0,len(X_CUR[0]),(1,1))).to(device)
            x_cur_t_new=X_CUR[[[0]],idx_x_cur_t_new]

            
            
        # if True:
        #     # the user is 100% sure that when he is absent or sleeping, the lamp
        #     # should be off
            
        #     if ((x_light_t_norm[:,:len(X_US[0])]).max(1)[1].item())==0:
        #         x_ls_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         x_cur_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         idx_x_cur_t_new=torch.tensor([[0]],device=device)

        #     if ((x_light_t_norm[:,:len(X_US[0])]).max(1)[1].item())==3:
        #         x_ls_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
        #         # x_cur_t_new=torch.tensor([[0]],dtype=torch.float32,device=device)
            
        # new states after having decided the proposed actions
        x_lr_t_new=LightService.getIndoorLight(x_ls_t_new,x_cur_t_new)
        
        x_us_t_new=x_us_t
        x_lr_t_new=x_lr_t_new
        x_le_t_new=x_le_t
        x_ls_t_new=x_ls_t_new
        x_cur_t_new=x_cur_t_new
        
        x_us_t_new_norm=OneHotEncoder._one_hot_encoder(X_US,x_us_t_new)
        x_lr_t_new_norm=x_lr_t_new/max_x_light
        x_le_t_new_norm=x_le_t_new/max_x_light
        x_light_t_new_norm=torch.cat((x_us_t_new_norm,
                             
                             x_le_t_new_norm),axis=1)

        # print(f'x_lr_t_new: {x_lr_t_new}')
        
        actions=(x_ls_t_new,x_cur_t_new)
        print(f'actions: {x_ls_t_new,idx_x_cur_t_new}')
        
        LightService.setStates(x_us_t_new,x_lr_t_new,x_le_t_new,x_ls_t_new,x_cur_t_new)
        reward=LightService.getReward(x_light_t_new_norm)
        print(f'x_lr_t_new: {x_lr_t_new}')


        totalEpochRewards[0,epoch]+=reward

        # print(f'reward: {reward}')
        reward_t=torch.tensor([[reward]],dtype=torch.float32,device=device)
        
        transition=(x_light_t_norm,actions,reward_t,x_light_t_new_norm)
        LightService.updateReplayMemory(transition)
        
        # train the neural network
        # if len(LightService.replayMemory)>=LightService.MinibatchSize:

        minibatch=LightService.train()

        if epoch%100==0 and epoch!=0:      
            # if step>=(steps-10):
            seenTrainingStates.append(minibatch)
                # seenTrainingStates=seenTrainingStates.append(minibatch)
        
        # redefine the current time step states
        x_lr_t=x_lr_t_new
        x_ls_t=x_ls_t_new
        x_cur_t=x_cur_t_new

        print(f'epoch:{epoch}, step:{step}, reward:{reward}, totalEpochRewards:{totalEpochRewards[0,epoch]}')

    

        
    if epoch%10==0:
        LightService.targetLightServiceModel.load_state_dict(LightService.LightServiceModel.state_dict())

    if epoch%100==0 and epoch!=0:      
        torch.save(LightService.LightServiceModel.state_dict(),dirName+f'/single_service_light_v1_{32+1+int(epoch/100)}_v7.pth')
        np.save(other_data_name+f'/seenTrainingStates_v1_{32+1+int(epoch/100)}_v7.npy',seenTrainingStates)
        np.save(other_data_name+f'/totalEpochRewards_v1_{32+1+int(epoch/100)}_v7.npy',totalEpochRewards)
        seenTrainingStates=deque(maxlen=100000)


torch.save(LightService.LightServiceModel.state_dict(),dirName+f'/single_service_light_v1_v7.pth')




# except Exception as e:

#     torch.save(LightService.LightServiceModel.state_dict(),dirName+f'/model_structure_2_v3_{7+1+int(epoch/100)}.pth')

# except KeyboardInterrupt:

#     torch.save(LightService.LightServiceModel.state_dict(),dirName+f'/model_structure_2_v3_{7+1+int(epoch/100)}.pth')

# except UserAbort:

#     torch.save(LightService.LightServiceModel.state_dict(),dirName+f'/model_structure_2_v3_{7+1+int(epoch/100)}.pth')