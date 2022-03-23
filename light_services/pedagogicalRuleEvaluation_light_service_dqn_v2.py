# Compared pedagogical method and neural network method by comparing data acquired from
# 'logicalRuleEvaluation.py' and 'generate_testing_dataSet.py (seen in enst cloud service)'
# %%
import torch 
import numpy as np 

import random
import os

# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

# %%
# create a data storing path

num_epochs=100
steps=288

random.seed(0)

dirName='11082021/model_save'
if not os.path.exists(dirName):
    os.makedirs(dirName)

other_data_name='11082021/section6_seen'
if not os.path.exists(other_data_name):
    os.makedirs(other_data_name)


# if os.path.exists("./../v3/"+other_data_name+"/pedagogicalRuleEvaluation_v2_48_with_seen.log"):
#     f = open("./../v3/"+other_data_name+"/pedagogicalRuleEvaluation_v2_48_with_seen.log", "r+")
# else:
#     f = open("./../v3/"+other_data_name+"/pedagogicalRuleEvaluation_v2_48_with_seen.log", "w")


# import logging

# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler("./../v3/"+other_data_name+"/pedagogicalRuleEvaluation_v2_48_with_seen.log", 'a'))
# print = logger.info

# %%
# load data for neural network
# totalEpochRewardsNN=np.load('./../v3/section6/not_seen_states'+'/totalEpochRewards_2_v3_with_not_seen_100.npy',allow_pickle=True)
# replayMemoryNN=np.load('./../v3/section6/not_seen_states'+'/replayMemory_2_v3_with_not_seen_100.npy',allow_pickle=True)

# # %%
# # load data for the rule system
# totalEpochRewardsRule=np.load('./../v3/section6/not_seen_states'+'/totalEpochRewards_2_v3_with_not_seen_RULES_100.npy',allow_pickle=True)
# replayMemoryRule=np.load('./../v3/section6/not_seen_states'+'/replayMemory_2_v3_with_not_seen_RULES_100.npy',allow_pickle=True)
# notSeenStatesRule=np.load('./../v3/section6/not_seen_states'+'/states_with_not_seen_RULES_100.npy',allow_pickle=True)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 10 epoch

# totalEpochRewardsNN=np.load('./../v3/section6/not_seen_states'+'/totalEpochRewards_2_v3_with_not_seen_100.npy',allow_pickle=True)
# replayMemoryNN=np.load('./../v3/section6/not_seen_states'+'/replayMemory_2_v3_with_not_seen_100.npy',allow_pickle=True)

# %%
# load data for the rule system
# totalEpochRewardsRule=np.load('./../v3/section6/not_seen_states'+'/totalEpochRewards_2_v3_with_not_seen_not_seen_RULES_100.npy',allow_pickle=True)
# replayMemoryRule=np.load('./../v3/section6/not_seen_states'+'/replayMemory_2_v3_with_not_seen_not_seen_RULES_100.npy',allow_pickle=True)
# notSeenStatesRule=np.load('./../v3/section6/not_seen_states'+'/states_with_not_seen_not_seen_RULES_100.npy',allow_pickle=True)

# %%
# %%
# %%
# load data for the rule system
# totalEpochRewardsRule=np.load('./../v3/section6/not_seen_decomp/totalEpochRewards_2_v3_with_not_seen_RULES_100.npy',allow_pickle=True)
# replayMemoryRule=np.load('./../v3/section6/not_seen_decomp/replayMemory_2_v3_with_not_seen_RULES_100.npy',allow_pickle=True)
# notSeenStatesRule=np.load('./../v3/section6/not_seen_decomp/states_with_not_seen_RULES_100.npy',allow_pickle=True)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# totalEpochRewardsNN=np.load('./../v3/'+other_data_name+'/totalEpochRewards_light_service_structure_2_v7_100_unseen.npy',allow_pickle=True)
# replayMemoryNN=np.load('./../v3/'+other_data_name+'/replayMemory_light_service_structure_2_v7_100_unseen.npy',allow_pickle=True)

# totalEpochRewardsRule=np.load('./../v3/'+other_data_name+'/totalEpochRewards_light_service_structure_2_v7_RULES_100_unseen.npy',allow_pickle=True)
# replayMemoryRule=np.load('./../v3/'+other_data_name+'/replayMemory_light_service_structure_2_v7_RULES_100_unseen.npy',allow_pickle=True)
# notSeenStatesRule=np.load('./../v3/'+other_data_name+'/notseen_states_light_service_structure_2_v7_RULES_100_unseen.npy',allow_pickle=True)


totalEpochRewardsNN=np.load('./../v3/'+other_data_name+'/totalEpochRewards_light_service_structure_2_v7_100_unseen.npy',allow_pickle=True)
replayMemoryNN=np.load('./../v3/'+other_data_name+'/replayMemory_light_service_structure_2_v7_100_unseen.npy',allow_pickle=True)

totalEpochRewardsRule=np.load('./../v3/'+other_data_name+'/totalEpochRewards_light_service_structure_2_v7_decomp_RULES_100_unseen.npy',allow_pickle=True)
replayMemoryRule=np.load('./../v3/'+other_data_name+'/replayMemory_light_service_structure_2_v7_decomp_RULES_100_unseen.npy',allow_pickle=True)
notSeenStatesRule=np.load('./../v3/'+other_data_name+'/notseen_states_light_service_structure_2_v7_decomp_RULES_100_unseen.npy',allow_pickle=True)


# calculate epoch rewards


def judgePos(notSeenStatesRule,i,epoch):
    if i<=notSeenStatesRule.shape[1]-2:
        if notSeenStatesRule[0,i+1]>=(epoch+1)*steps:
            return True
    elif i==notSeenStatesRule.shape[1]-1:
         
        return True
    
    return False
    
seenEpochRewards=np.full([1,num_epochs],288)*4
idx_seen=np.full([1,num_epochs],None)

if notSeenStatesRule[0,0]!=None:
    totalEpochRewardsNN=np.full([1,num_epochs],0)

num=0
num_delete=0
for epoch in range(num_epochs):
    print(f'epoch: {epoch}')
    rewards=0
    restIdx=np.array([[i for i in range(epoch*steps,(epoch+1)*steps)]]).reshape(1,-1)
 
    idx_seen[0,epoch]=restIdx
    toDelete=np.full([1,1],None)

    if notSeenStatesRule[0,0]!=None:

        for i in range(notSeenStatesRule.shape[1]):
            
            if notSeenStatesRule[0,i]>=(epoch*steps) and notSeenStatesRule[0,i]<((epoch+1)*steps):
         
                if toDelete[0,0]==None:
                    toDelete[0,0]=notSeenStatesRule[0,i]-epoch*steps
                    toDelete=toDelete.astype(np.int)
                else:
                    toDeleteInt=np.array([[notSeenStatesRule[0,i]-epoch*steps]]).astype(np.int)
                    toDelete=np.concatenate((toDelete,toDeleteInt),axis=1)



            
            print(f'i: {notSeenStatesRule[0,i]}')
            result=judgePos(notSeenStatesRule,i,epoch)
                
            if result:
                if toDelete[0,0]!=None:
                    idx_seen[0,epoch]=np.delete(idx_seen[0,epoch],toDelete,axis=1)
                    num_not_seen=toDelete.shape[1]
                    seenEpochRewards[0,epoch]-=num_not_seen*4
                    num_delete+=toDelete.shape[1]
                    total=restIdx.shape[1]+toDelete.shape[1]
              
                    if total!=288:
                        print('================================================================')

                num=num+restIdx.shape[1]
     
                for idx in idx_seen[0,epoch][0]:
           
                    totalEpochRewardsNN[0,epoch]+=replayMemoryNN[idx,2].item()
           
         
                break

print(f'idx_not_seen num: {num_delete}')
print(f'idx_seen num: {num}')
if notSeenStatesRule[0,0]!=None:
    replayMemoryNN=np.delete(replayMemoryNN,notSeenStatesRule,axis=0)
print(f'replayMemoryNN shape: {replayMemoryNN.shape}')
print(f'replayMemoryRule shape: ',{replayMemoryRule.shape})

idx_seen_2=np.array([[i for i in range(0,100*288)]]).reshape(1,-1)
if notSeenStatesRule[0,0]!=None:
    idx_seen_2=np.delete(idx_seen_2,notSeenStatesRule,axis=1)
print(f'idx_seen_2 shape: {idx_seen_2.shape}')
print(f'notSeenStatesRule shape: {notSeenStatesRule.shape}')
print(f'notSeenStatesRule shape: {notSeenStatesRule}')


# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")

# print("================================================================================================")
# print("================================================================================================")
# print("================================================================================================")
print("================================================================================================")
print("================================================================================================")

print("================================================================================================")
num=0
for i in range(replayMemoryNN.shape[0]):
    # print('current step: ',i)
    if (replayMemoryNN[i,0]==replayMemoryRule[i,0]).all():
        if (replayMemoryNN[i,1][0]==replayMemoryRule[i,1][0]) and (replayMemoryNN[i,1][1]==replayMemoryRule[i,1][1]):
            num+=1

print(num)
print(f'replayMemoryNN shape: {replayMemoryNN.shape}')
         
# np.save('./../v3/section6/not_seen_states'+f'/totalEpochRewards_2_v3_with_not_seen_NN_100.npy',totalEpochRewardsNN)  
# np.save('./../v3/section6/not_seen_states'+f'/replayMemory_2_v3_with_not_seen_NN_100.npy',replayMemoryNN)  
# np.save('./../v3/section6/not_seen_states'+f'/seenEpochRewards_100.npy',seenEpochRewards) 
# np.save('./../v3/section6/not_seen_states'+f'/idx_seen_100.npy',idx_seen) 
# np.save('./../v3/section6/not_seen_states'+f'/idx_seen_100_2.npy',idx_seen_2)  


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 10 epochs


# np.save('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_2_v7_NN_100_unseen.npy',totalEpochRewardsNN)  
# np.save('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_2_v7_NN_100_unseen.npy',replayMemoryNN)  
# np.save('./../v3/'+other_data_name+f'/seenEpochRewards_light_service_structure_2_v7_100_unseen.npy',seenEpochRewards) 
# np.save('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_100_unseen.npy',idx_seen) 
# np.save('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_100_2_unseen.npy',idx_seen_2)  

np.save('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_2_v7_NN_100_decomp_unseen.npy',totalEpochRewardsNN)  
np.save('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_2_v7_NN_100_decomp_unseen.npy',replayMemoryNN)  
np.save('./../v3/'+other_data_name+f'/seenEpochRewards_light_service_structure_2_v7_100_decomp_unseen.npy',seenEpochRewards) 
np.save('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_100_decomp_unseen.npy',idx_seen) 
np.save('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_100_2_decomp_unseen.npy',idx_seen_2)  



