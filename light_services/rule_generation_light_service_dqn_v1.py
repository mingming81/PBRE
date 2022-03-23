import torch
# import numpy as np

# Pedagogical rule extraction methon
# %%
import random

import os

import numpy as np

from sample_enumerate_abstraction_pedagogical_ing_single_light_service_dqn_v1 import *

random.seed(0)

torch.autograd.set_detect_anomaly(True)

# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

# %%
# create a data storing path

dirName='11082021/model_save'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='11082021/section6_seen'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)

print('beginnnnnnnnnnnnnnnnn')
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
num_epochs=100
steps=288

learning_rate=0.001

epsilon=0.1

max_x_light=1005


# %%

# totalEpochRewards=np.load('./../v3/section6/seen_states'+'/totalEpochRewards_2_v3_19.npy',allow_pickle=True)[0,:288*100]

# # len(replayMemory): 24000
# replayMemory=np.load('./../v3/section6/seen_states/seenTrainingStates_2_v3_19.npy',allow_pickle=True).reshape(-1,4)
# replayMemory=replayMemory[:288*100,:]

# print(replayMemory.shape)
# import sys
# sys.exit()



# %%
# totalEpochRewards.shape: 1*1000
# totalEpochRewards=np.load('./../v3/section6/not_seen_states'+'/totalEpochRewards_2_v3_with_not_seen_100.npy',allow_pickle=True)

# # len(replayMemory): 24000
# replayMemory=np.load('./../v3/section6/not_seen_states'+'/replayMemory_2_v3_with_not_seen_100.npy',allow_pickle=True)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 10epochs

totalEpochRewards=np.load('./../v3/'+other_data_name+'/totalEpochRewards_light_service_structure_2_v9_100_seen.npy',allow_pickle=True)

replayMemory=np.load('./../v3/'+other_data_name+'/replayMemory_light_service_structure_2_v9_100_seen.npy',allow_pickle=True)


# %%
class ServiceModel(torch.nn.Module):
    def __init__(self):
        super(ServiceModel,self).__init__()
        
        num_input_features=4

        num_hidden=10
       
        num_output_features=8

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(num_input_features,num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,num_output_features)

            )

        # self.hidden=torch.nn.functional.relu(num_input_features,num_hidden)
        
        
        # self.output=torch.nn.Linear(num_hidden,num_output_features)
        
        
    def forward(self,x):
        x = self.flatten(x)
        y=self.linear_relu_stack(x)

        return y







    
if __name__ == "__main__":

    model=ServiceModel().to(device)
    model.load_state_dict(torch.load('./../v3/'+dirName+f'/single_service_light_v3_55_v9.pth'))
    
    # %%
    # minibatch=replayMemory
    # lists
    startStates=[transition[0] for transition in replayMemory]
    # print(startStates)
    # lists
    endStates=[transition[3] for transition in replayMemory]
    # %%
    seapeda=Seapeda()
    # %%
    
    numRefineTree=0
    ruleSet=np.full([1,4],None)
    lenX=len(startStates)
    print(f'lenX: {lenX}')

    # import sys

    # sys.exit()
    
    for i in range(lenX):
        
        if i==241:
            print('yes')
        
        print(f'state idx: {i}')
        # import sys
        # sys.exit()
        
        x=startStates[i]
        
        x=x.to(device)
        
        
        rule=seapeda.ruleExtract(x,model)
        # print('extract rule:\n ',rule)
        # print("\n===========================\n")
        
        seapeda.update(rule)
        
        
        # rule structure: fullSet: np.array([[
        #                        np.array([[np.array([[str:ls,min,max]]),int:counts]]),
        #                        np.array([[np.array([[str:cur,min,max]]),int:counts]])
        #                                                                           ]])
            

        
    rule=seapeda.refineRule()
        # print('refined rule:\n ',rule)
        # print("\n===========================\n")
        
        
        
        # if ruleSet[0,0]==None:
    ruleSet=copy.deepcopy(rule)
        
        # np.savetxt('rule.txt',ruleSet,fmt='%1.4s')
        
        # print(f'total rule:\n ',ruleSet)
        
        # if i==600:
        #     print(ruleSet)
            
        # num_rule=ruleSet.shape[0]
        # num=0
        # for rule in range(num_rule):
        #     # if ruleSet[rule,2][0,0][0,1]==2:
                
        #     print('\n ',ruleSet[rule,:])
            
        # print("\n===========================\n")
                # num+=1
        
        # if num==0 and i==0:
        #     print(i)
        #     pass
                
        
        # else:
        #     ruleSet=np.concatenate((ruleSet,rule),axis=0)
        
        # numRefineTree+=1
        
        # every 24*5 states, update one time the tree structure.
        # if numRefineTree==24*5:
        #     # replace original ruleTree after having obtained refine rule using refineRule()
        #     seapeda.updateRefinedTree(rule)

        # if i%5000==0 and i!=0:
        #     np.save('./../v3/section6/not_seen_states'+f'/logical_rules_with_not_seen_100_{int(i/1000)}_5.npy',ruleSet)        
    
    # np.save('./../v3/section6/not_seen_states'+'/logical_rules_with_not_seen_100.npy',ruleSet)

    # %%%%%%%%%%%%%%%%%%%%%%%%%
    # 10 epochs

    np.save('./../v3/'+other_data_name+'/logical_rules_light_service_structure_2_v9_100.npy',ruleSet)
    print(f'ruleSet shape: {ruleSet.shape}')


    
        
        
        
        
        

        
        
        
        
        
        
        
        
        
