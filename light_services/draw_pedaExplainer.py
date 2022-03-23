import numpy as np 


# %%
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
mpl.style.use('seaborn-white')
import matplotlib.ticker as mtick

# plt.rcParams['axes.xmargin'] = 0
# plt.xlim(left=0)
epochs=100
x=np.arange(0,100).reshape(1,-1)
fullEpoch=np.ones((1,100))*288*4
# %%
# draw similarities



dirName='11082021/model_save'

other_data_name='11082021/section6_seen'

# %%
# totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_mlt_3_v1_100.npy',allow_pickle=True)
# totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_100_mlt_3.npy',allow_pickle=True) #lstm_multi_services_3.py
# totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_100_mlt.npy',allow_pickle=True)  #lstm_multi_services.py
# totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_100_mlt_4.npy',allow_pickle=True)[:100,:]  #lstm_multi_services_4.py
# totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_mlt_5_v1_100.npy',allow_pickle=True)[:100,:]  #lstm_multi_services_5_v1.py
# totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_mlt_22_100.npy',allow_pickle=True)[:100,:]  #lstm_multi_services_22.py
# totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_mlt_23_100.npy',allow_pickle=True)[:100,:]  #lstm_multi_services_23.py
totalEpoch=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_lstm_v1_mlt_24_100.npy',allow_pickle=True)[:100,:]  #lstm_multi_services_24.py




x=np.arange(0,100).reshape(1,-1)
fig,ax = plt.subplots(figsize = (12,7))

ax.plot(x[0,:],totalEpoch[:,0],label='light service total rewards')
ax.plot(x[0,:],totalEpoch[:,1],label='temperature service total rewards')
ax.plot(x[0,:],totalEpoch[:,2],label='air service total rewards')
# ax.plot(x[0,:],totalEpoch[:,0]+totalEpoch[:,1]+totalEpoch[:,2],label='curtain priority total rewards')
# ax.plot(x[0,:],totalEpoch[:,1]+totalEpoch[:,2],label='window priority total rewards')
# ax.plot(x[0,:],totalEpoch[:,1]+totalEpoch[:,2],label='open time priority total rewards')

plt.setp(ax.get_xticklabels(), fontsize=25)
plt.setp(ax.get_yticklabels(), fontsize=25)

l=[0,20,40,60,80,100]
labels = [0,20,40,60,80,100]

ax.set_xticks(l)

ax.set_xticklabels(labels)
ax.xmargin=0

# ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

plt.legend(fontsize=20,frameon=True,framealpha=0.3,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

ax.set_xlabel('epoch',fontsize=30)
ax.set_ylabel('similarity',fontsize=30)

fig.tight_layout()
plt.margins(x=0.02)

plt.show()



# %%
# pedaExplainer

# idx_seen_100=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_lstm_100_seen.npy',allow_pickle=True)
# idx_seen_100_2=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_lstm_100_2_seen.npy',allow_pickle=True)
# notSeenStatesRule=np.load('./../v3/'+other_data_name+f'/notseen_states_light_service_structure_lstm_RULES_100_seen.npy',allow_pickle=True)

# replayMemoryRule=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_lstm_RULES_100_seen.npy',allow_pickle=True)
# replayMemoryNN=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_lstm_NN_100_seen.npy',allow_pickle=True)
# %%
idx_seen_100=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_lstm_100_unseen.npy',allow_pickle=True)
idx_seen_100_2=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_lstm_100_2_unseen.npy',allow_pickle=True)
notSeenStatesRule=np.load('./../v3/'+other_data_name+f'/notseen_states_light_service_structure_lstm_RULES_100_unseen.npy',allow_pickle=True)

replayMemoryRule=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_lstm_RULES_100_unseen.npy',allow_pickle=True)
replayMemoryNN=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_lstm_NN_100_unseen.npy',allow_pickle=True)

# %%
# decomp
# idx_seen_100=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_p3_100_decomp_seen.npy',allow_pickle=True)
# idx_seen_100_2=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_p3_100_2_decomp_seen.npy',allow_pickle=True)
# notSeenStatesRule=np.load('./../v3/'+other_data_name+f'/notseenStates_light_service_structure_2_v7_p3_RULES_100_decomp_seen.npy',allow_pickle=True)

# replayMemoryRule=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_2_v7_p3_RULES_100_decomp_seen.npy',allow_pickle=True)
# replayMemoryNN=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_2_v7_p3_NN_100_decomp_seen.npy',allow_pickle=True)

# %%

# idx_seen_100=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_p3_100_decomp_unseen.npy',allow_pickle=True)
# idx_seen_100_2=np.load('./../v3/'+other_data_name+f'/idx_light_service_structure_2_v7_p3_100_2_decomp_unseen.npy',allow_pickle=True)
# notSeenStatesRule=np.load('./../v3/'+other_data_name+f'/notseenStates_light_service_structure_2_v7_p3_RULES_100_decomp_unseen.npy',allow_pickle=True)

# replayMemoryRule=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_2_v7_p3_RULES_100_decomp_unseen.npy',allow_pickle=True)
# replayMemoryNN=np.load('./../v3/'+other_data_name+f'/replayMemory_light_service_structure_2_v7_p3_NN_100_decomp_unseen.npy',allow_pickle=True)


# %%%%%%%%%%%%%%
# similarity calculation for action proposition

countSame=0

# np.save('./../v3/section6'+'/num_sameStates_100.npy',num_sameStates)
# np.save('./../v3/section6'+'/num_totalSeenState_100.npy',num_totalSeenState)
# num_sameStates=np.load('./../v3/section6'+'/num_sameStates_100.npy',allow_pickle=True)
# num_totalSeenState=np.load('./../v3/section6'+'/num_totalSeenState_100.npy',allow_pickle=True)

num_totalSeenState=np.full([1,epochs],None)
num_sameStates=np.full([1,epochs],None)


st=0
for epoch in range(epochs):
    num=0
    num_len=idx_seen_100[0,epoch].shape[1]
    num_totalSeenState[0,epoch]=num_len
    num_len=num_len+st
    for i in range(st,num_len):
        # print(i)
        if (replayMemoryNN[i,0]==replayMemoryRule[i,0]).all():
            if (replayMemoryNN[i,1][0]==replayMemoryRule[i,1][0]) and (replayMemoryNN[i,1][1]==replayMemoryRule[i,1][1]):
                num+=1
            else:
                print('yes')
    num_sameStates[0,epoch]=num
    st=st+idx_seen_100[0,epoch].shape[1]

pct=num_sameStates/num_totalSeenState

x=np.arange(0,100).reshape(1,-1)
fullEpoch=np.ones((1,100))*288*4



fig,ax = plt.subplots(figsize = (7,4))
# fig,ax=plt.subplots()


################## 
# section6: unseenStates+evaluation2 (prediction performance)+peda

ax.plot(x[0,:],pct[0,:]*100,label='similarity',color='tab:blue')


################## 

plt.setp(ax.get_xticklabels(), fontsize=25)
plt.setp(ax.get_yticklabels(), fontsize=25)

l=[0,20,40,60,80,100]
labels = [0,20,40,60,80,100]

ax.set_xticks(l)

ax.set_xticklabels(labels)
ax.xmargin=0

ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))


# ax.set_xticks(x[0,:-1])
# ax.set_xticklabels(x[0,:50])

# ax.legend(['sum', 'extracted rules','neural network'],loc=1,fontsize=11)

plt.legend(fontsize=20,ncol=3,frameon=True,framealpha=0.3,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

ax.set_xlabel('epoch',fontsize=30)
ax.set_ylabel('similarity',fontsize=30)
# plt.grid()
fig.tight_layout()
plt.margins(x=0.02)
# plt.savefig('figs/performance_rl_nn.eps', format='eps')

plt.show()

# %%

# pct of predicted rules

# seenEpochRewards_with_not_seen=np.load('./../v3/'+other_data_name+f'/seenEpochRewards_light_service_structure_lstm_100_seen.npy')
# seenStepNums=(seenEpochRewards_with_not_seen/4).astype(np.int32)
# seenStepNums=(seenEpochRewards_with_not_seen)

# totalEpochRewards_2_v3_with_not_seen_nn=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_lstm_NN_100_seen.npy')

# totalEpochRewards_2_v3_with_not_seen_rules=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_lstm_RULES_100_seen.npy')

# %%
seenEpochRewards_with_not_seen=np.load('./../v3/'+other_data_name+f'/seenEpochRewards_light_service_structure_lstm_100_unseen.npy')
seenStepNums=(seenEpochRewards_with_not_seen/4).astype(np.int32)
seenStepNums=(seenEpochRewards_with_not_seen)

totalEpochRewards_2_v3_with_not_seen_nn=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_lstm_NN_100_unseen.npy')

totalEpochRewards_2_v3_with_not_seen_rules=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_lstm_RULES_100_unseen.npy')


# %%
# # # decomp
# seenEpochRewards_with_not_seen=np.load('./../v3/'+other_data_name+f'/seenEpochRewards_light_service_structure_2_v7_p3_100_decomp_seen.npy')
# seenStepNums=(seenEpochRewards_with_not_seen/4).astype(np.int32)
# seenStepNums=(seenEpochRewards_with_not_seen)

# totalEpochRewards_2_v3_with_not_seen_nn=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_2_v7_p3_NN_100_decomp_seen.npy')

# totalEpochRewards_2_v3_with_not_seen_rules=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_2_v7_p3_RULES_100_decomp_seen.npy')

# %%
# seenEpochRewards_with_not_seen=np.load('./../v3/'+other_data_name+f'/seenEpochRewards_light_service_structure_2_v7_p3_100_decomp_unseen.npy')
# seenStepNums=(seenEpochRewards_with_not_seen/4).astype(np.int32)
# seenStepNums=(seenEpochRewards_with_not_seen)

# totalEpochRewards_2_v3_with_not_seen_nn=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_2_v7_p3_NN_100_decomp_unseen.npy')

# totalEpochRewards_2_v3_with_not_seen_rules=np.load('./../v3/'+other_data_name+f'/totalEpochRewards_light_service_structure_2_v7_p3_RULES_100_decomp_unseen.npy')



# %%
seenStepNums=(seenEpochRewards_with_not_seen/4).astype(np.int32)
seenStepNums=(seenEpochRewards_with_not_seen)

import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize = (7,4))
# fig,ax=plt.subplots()

################## 
# section6: unseenStates+evaluation3 (prediction performance)+peda
################## 
width = 0.25
# ax.bar(x[0,:50]-width,seenEpochRewards_with_not_seen[0,:50],width=width,color='r')
# ax.bar(x[0,:50],totalEpochRewards_2_v3_with_not_seen_rules[0,:50],width=width,color='yellow')
# ax.bar(x[0,:50]+width,totalEpochRewards_2_v3_with_not_seen_nn[0,:50],width=width,color='b')

# ax.plot(x[0,:],seenEpochRewards_with_not_seen[0,:],'r',label='full',linewidth=3, alpha=0.8)
# ax.plot(x[0,:],totalEpochRewards_2_v3_with_not_seen_nn[0,:],'bo--',label='nn')
# ax.plot(x[0,:],totalEpochRewards_2_v3_with_not_seen_rules[0,:],color='g',linewidth=2,label='rules')


################## 
# section6: unseenStates+evaluation2 (prediction performance)+peda
# ax.plot(x[0,:-1],seenStepNums[0,:-1]/4/288,label='number of rules predicted steps')
ax.plot(x[0,:],seenStepNums[0,:]/4/288*100,label="pct of inferable steps by rules",color='tab:blue')


# ax.plot(totalEpochRewards_2_v3_with_not_seen_nn[0,:]-totalEpochRewards_2_v3_with_not_seen_rules[0,:])

################## 

plt.setp(ax.get_xticklabels(), fontsize=25)
plt.setp(ax.get_yticklabels(), fontsize=25)

# l=[0,20,40,60,80,100]
# labels = [0,20,40,60,80,100]

ax.set_xticks(l)

ax.set_xticklabels(labels)
ax.xmargin=0

ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))


# ax.set_xticks(x[0,:-1])
# ax.set_xticklabels(x[0,:50])

# ax.legend(['sum', 'extracted rules','neural network'],loc=1,fontsize=11)

plt.legend(fontsize=18,ncol=3,frameon=True,framealpha=0.3,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

ax.set_xlabel('epoch',fontsize=30)
ax.set_ylabel('pct of inferable \nstates by rules',fontsize=30)
# ax.set_ylabel('Reward',fontsize=30)
# plt.grid()
fig.tight_layout()
plt.margins(x=0.02)
# plt.savefig('figs/performance_rl_nn.eps', format='eps')

plt.show()
# %%

# reward

import matplotlib.pyplot as plt


# fig,ax=plt.subplots()

import matplotlib.style
import matplotlib as mpl
from cycler import cycler
# mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
mpl.style.use('seaborn-white')
import matplotlib.ticker as mtick


seenEpochRewards_with_not_seen=np.zeros((1,100))

for i in range(replayMemoryNN.shape[0]):
    
    seenEpochRewards_with_not_seen[0,int(i/288)]+=abs(replayMemoryNN[i,2].item())
    
    

# %%
fig,ax = plt.subplots(figsize = (7,4))
################## 
# section6: unseenStates+evaluation3 (prediction performance)+peda
################## 
width = 0.25
# ax.bar(x[0,:50]-width,seenEpochRewards_with_not_seen[0,:50],width=width,color='r')
# ax.bar(x[0,:50],totalEpochRewards_2_v3_with_not_seen_rules[0,:50],width=width,color='yellow')
# ax.bar(x[0,:50]+width,totalEpochRewards_2_v3_with_not_seen_nn[0,:50],width=width,color='b')

ax.plot(x[0,:],seenEpochRewards_with_not_seen[0,:],'r',label='total reward',linewidth=3, alpha=0.8)
ax.plot(x[0,:],totalEpochRewards_2_v3_with_not_seen_nn[0,:],'bo--',label='ARL')
ax.plot(x[0,:],totalEpochRewards_2_v3_with_not_seen_rules[0,:],'g',linewidth=2,label='rules')


################## 
# section6: unseenStates+evaluation2 (prediction performance)+peda
# ax.plot(x[0,:-1],seenStepNums[0,:-1]/4/288,label='number of rules predicted steps')
# ax.plot(x[0,:],seenStepNums[0,:]/4/288*100,label="pct of predictable steps by rules",color='tab:blue')


# ax.plot(totalEpochRewards_2_v3_with_not_seen_nn[0,:]-totalEpochRewards_2_v3_with_not_seen_rules[0,:])

################## 

plt.setp(ax.get_xticklabels(), fontsize=25)
plt.setp(ax.get_yticklabels(), fontsize=25)

# l=[0,20,40,60,80,100]
# labels = [0,20,40,60,80,100]

ax.set_xticks(l)

ax.set_xticklabels(labels)
ax.xmargin=0

# ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))


# ax.set_xticks(x[0,:-1])
# ax.set_xticklabels(x[0,:50])

# ax.legend(['sum', 'extracted rules','neural network'],loc=1,fontsize=11)

plt.legend(fontsize=18,ncol=3,frameon=True,framealpha=0.3,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

ax.set_xlabel('epoch',fontsize=30)
ax.set_ylabel('Reward',fontsize=30)

fig.tight_layout()
plt.margins(x=0.02)


plt.show()

