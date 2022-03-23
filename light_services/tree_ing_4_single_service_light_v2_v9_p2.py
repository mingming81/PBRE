import torch 

import numpy as np

import sys

import copy

from refine_rules import *

# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device=torch.device("cpu")


# import os

# if os.path.exists("v3/section6/not_seen_states/single_service_light_v2_48.log"):
#     f = open("v3/section6/not_seen_states/single_service_light_v2_48.log", "r+")
# else:
#     f = open("v3/section6/not_seen_states/single_service_light_v2_48.log", "w")


# import logging

# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler('v3/section6/not_seen_states/single_service_light_v2_48.log', 'a'))
# print = logger.info


print('yo!')
print(f"=3=======================================================")
print(f"==3======================================================")
print(f"===3=====================================================")

print(f"======3==================================================")
print(f"=======3=================================================")
print(f"========3================================================")
print(f"=========3===============================================")
print(f"==========3==============================================")
print(f"===========3=============================================")
print(f"============3============================================")
print(f"=============3===========3================================")
print(f"==============3==========================================")
print(f"===============3=========================================")
print(f"================3========================================")
print(f"=================3=======================================")

print(f"==================3======================================")
print(f"===================3=====================================")
print(f"====================3====================================")
print(f"=====================3===================================")
print(f"======================3==================================")
print(f"=======================3=================================")
print(f"========================3================================")
print(f"==========================3==============================")
print(f"===========================3=============================")
print(f"============================3============================")
print(f"=============================3===========================")
print(f"==============================3==========================")

print(f"===============================3=========================")
print(f"================================3========================")
print(f"=================================3=======================")
print(f"==================================3======================")
print(f"===================================3=====================")
print(f"====================================3====================")
print(f"=====================================3===================")
print(f"======================================3==================")
print(f"=======================================3=================")
max_x_light=1005





# %%
# check None in np array wit shape (1,n)
def checkNone(x):
    for i in range(x.shape[1]):
        if x[0,i] is None:
            return True
    return False
    
# %%
# nearNodes: np.array([[[int:count,torch:min,torch:max,subNodes:]],
#                     [[[int:count,torch:min,torch:max,subNodes:]],
#                     [[[int:count,torch:min,torch:max,subNodes:]],
#                                   ...]])
# rule: np.array([[str:'the state is ...', min, max]])

# resultRange: np.array([[str,torch:min,torch:max,int:count,subNodes:]],
#                        [[str,torch:min,torch:max,int:count,subNodes:]],
#                        [[str,torch:min,torch:max,int:count,subNodes:]],
#                        [[str,torch:min,torch:max,int:count,subNodes:]],
#                                      ...])
def nodeRange(nearNodes,rule):
    resultRange=np.full([1,5],None)
    
    lenNearNode=nearNodes.shape[0]
    for i in range(lenNearNode):
        
        resultRangeInt=np.full([1,5],None)
        
        if nearNodes[i][1]>=rule[0,1] and nearNodes[i][2]>=rule[0,2]:
            resultRangeInt[0,0]=rule[0,0]
            resultRangeInt[0,1]=rule[0,1]
            resultRangeInt[0,2]=nearNodes[i][2]
            resultRangeInt[0,3]=nearNodes[i][0]+1
            resultRangeInt[0,4]=nearNodes[i][3]
            
        elif nearNodes[i][1]<=rule[0,1] and nearNodes[i][2]>=rule[0,2]:
            resultRangeInt[0,0]=rule[0,0]
            resultRangeInt[0,1]=nearNodes[i][1]
            resultRangeInt[0,2]=nearNodes[i][2]
            resultRangeInt[0,3]=nearNodes[i][0]+1
            resultRangeInt[0,4]=nearNodes[i][3]
        
        elif nearNodes[i][1]<=rule[0,1] and nearNodes[i][2]<=rule[0,2]:
            resultRangeInt[0,0]=rule[0,0]
            resultRangeInt[0,1]=nearNodes[i][1]
            resultRangeInt[0,2]=rule[0,2]
            resultRangeInt[0,3]=nearNodes[i][0]+1
            resultRangeInt[0,4]=nearNodes[i][3]
                        
        if resultRange[0,0]==None:
            resultRange=copy.deepcopy(resultRangeInt)
        else:
            resultRange=np.concatenate((resultRange,resultRangeInt),axis=0)
    
    return resultRange



# %%


def sortRules(fullSet):
    
    # fullSet=fullSet.reshape(-1,4)
    
    fullSet=sorted(fullSet,key=lambda x: x[3][0,0][0,1])
    
    fullSet=np.array(fullSet)
    
    # combination
    
    
    
    return fullSet

# %%

def combineRules(fullSet1):
    
    fullSet1=sortRules(fullSet1)
    
    combined=fullSet1[0]

    fullSet2=np.array([[None]])
        
    
    
    for i in range(1,fullSet1.shape[0]):
        
        toCombined=fullSet1[i]
        
        if toCombined[3][0,0][0,1]<=combined[3][0,0][0,2]:
            
            combined[3][0,0][0,2]=max(toCombined[3][0,0][0,2],combined[3][0,0][0,2])
            
        else:
            if fullSet2[0,0]==None:
                
                fullSet2=copy.deepcopy(combined.reshape(-1,4))
                
            else:
                
                fullSet2=np.concatenate((fullSet2,combined.reshape(-1,4)),axis=0)
            
            combined=fullSet1[i]
            
            # if i==fullSet1.shape[0]-1:
            #     fullSet2=np.concatenate((fullSet2,toCombined.reshape(-1,4)),axis=0)
                
    if fullSet2[0,0]==None:
                
        fullSet2=copy.deepcopy(combined.reshape(-1,4))
        
    else:
        
        fullSet2=np.concatenate((fullSet2,combined.reshape(-1,4)),axis=0)
                
    return fullSet2
   

# %%
# insert data should be a numpy array
class Node:
    def __init__(self,data):
        self.data=data
        self.subNodes=np.full([1,2], None)
       
    def insert(self,rule,i):
        assert rule.shape==(1,3), "the check state should have shape (1,3)"
        self.subNodes[i][0]=Node(rule)
        
 
    # rule: torch([[str:, torch:min, torch:max]])
    def checkNode(self,rule,checkType):
        
        case=None
        
        # check only the subnode of the current root
        assert self.subNodes[0][0]!=None, "No subNodes are defined, use Node.insert(data) to create subNodes"
        assert rule.shape==(1,3), "the check state should have shape (1,3) containing the description, the min value and the max value"
        
        length=self.subNodes.shape[0]
        
        for node in range(length):
            # min==min, max==max
            if checkType==0 or checkType==1 or checkType==2:
                
                if self.subNodes[node][0].data[0,1]==rule[0,1] or self.subNodes[node][0].data[0,2]==rule[0,2]:
                    case=0
                    # return index of subnodes, category of the same state (lamp state), the truth of whether having the same states
                    self.subNodes[node][1]+=1
                    
                    # return the index
                    return (node,node),True
        
        # int:count,tensor:min,tensor:max
        nearNodes=np.full([1,4],None)
        toDeleteNodes=np.full([1,1],None)
        for node in range(length):
            # ensure that the compared states only support continuous states
            if checkType!=0 and checkType!=1 and checkType!=2:
                # this case is responsible for changing the value too, patience equal 
                if (abs(self.subNodes[node][0].data[0,1]-rule[0,1])<=torch.tensor([[0.001]])*1000 or abs(self.subNodes[node][0].data[0,2]-rule[0,2])<=torch.tensor([[0.001]])*1000) or (rule[0,1]>=self.subNodes[node][0].data[0,1] and rule[0,1]<=self.subNodes[node][0].data[0,2]):
                    case=1
                    if toDeleteNodes[0,0]==None:
                        toDeleteNodes[0,0]=node
                    else:
                        toDeleteNodes=np.concatenate((toDeleteNodes,np.array([[node]])),axis=1)
                        
                    if nearNodes[0][0]==None:
                        nearNodes[0][0]=self.subNodes[node][1]  # int: count
                        nearNodes[0][1]=self.subNodes[node][0].data[0,1] #torch:min
                        nearNodes[0][2]=self.subNodes[node][0].data[0,2] #torch:max
                        nearNodes[0][3]=self.subNodes[node][0].subNodes  #torch: subNodes
                        
                    else:
                        nearNodeInter=np.full([1,4],None)
                        nearNodeInter[0][0]=self.subNodes[node][1] # int: count
                        nearNodeInter[0][1]=self.subNodes[node][0].data[0,1] #torch:min
                        nearNodeInter[0][2]=self.subNodes[node][0].data[0,2] #torch:max
                        nearNodeInter[0][3]=self.subNodes[node][0].subNodes #torch: subNodes
                        
                        nearNodes=np.concatenate((nearNodes,nearNodeInter), axis=0)
                        
        
        if case==1:
            
            self.subNodes=np.delete(self.subNodes,toDeleteNodes.astype(np.int),axis=0)
            
            # [[str,torch:min,torch:max,int:count,subNodes:]]
            resultRange=nodeRange(nearNodes,rule)
            
            startPoint=self.subNodes.shape[0]
            # append 
            lenResultRange=resultRange.shape[0]
            for i in range(lenResultRange):
                subNodeInt=np.full([1,2],None)
                resultRangeInt=np.full([1,3],None)
                resultRangeInt[0,0]=resultRange[i][0]
                resultRangeInt[0,1]=resultRange[i][1]
                resultRangeInt[0,2]=resultRange[i][2]
                # define the value for a subNode
                subNodeInt[0,0]=Node(resultRangeInt)
                # define subNodes for a subNode
                subNodeInt[0,0].subNodes=resultRange[i][4]
                # define a counter for a subNode
                subNodeInt[0,1]=resultRange[i][3]
                self.subNodes=np.concatenate((self.subNodes,subNodeInt),axis=0)
            
            endPoint=self.subNodes.shape[0]
    
            return (startPoint,endPoint-1),True
        
        else:
            startPoint=self.subNodes.shape[0]
            subNodeInt=np.full([1,2],None)
            resultRangeInt=np.full([1,3],None)
            
            resultRangeInt[0,0]=rule[0,0]
            resultRangeInt[0,1]=rule[0,1]
            resultRangeInt[0,2]=rule[0,2]
            subNodeInt[0,0]=Node(resultRangeInt)
            subNodeInt[0,1]=1
            self.subNodes=np.concatenate((self.subNodes,subNodeInt),axis=0)
            
            endPoint=self.subNodes.shape[0]
            return (startPoint,endPoint-1),True
            
    
    def update(self,toCheck,checkType=0,add=0):
        checkType=checkType+add
        lengthNode0,lengthNode1,lengthNode2,lengthNode3=None,None,None,None
        
        
        if checkType==3:
            return
        
        # if checkType==2:
        #     # print("verifying....")
        #     pass

        
        # if lengthNode0==1 and lengthNode1==1 and lengthNode2==4 and lengthNode3==1 and lengthNode4==4:
        #     print('yes')
        
        # toCheck: np.array([rule_lamp,rule_curtain,rule_user_state,rule_indoor_light,rule_outdoor_light])
        # rule: np.array([str:'the lamp is off', torch.tensor: torch.tensor([[0]])])
        # totalLevel=toCheck.shape[1]
        rule=toCheck[0,checkType]
        
        rule1=rule[0,1]
        # rule: np.array([str:'the lamp is off', torch.tensor (min): torch.tensor([[0]]),torch.tensor(max): torch.tensor([[0]])])
        
        # if checkType==0:
        #     if rule1==4:
        #         # print("veryfying...")
        #         pass
            
        rule=np.concatenate((rule,rule1),axis=1)
        rule[0,2]=torch.tensor([[rule[0,2]]])
        # print(f'rule: {rule}')
        
        
        # print(self.subNodes[-1][0])
        # print('\n')
        
        if self.subNodes[-1][0]==None:
            self.insert(rule,-1)
            self.subNodes[-1][1]=1
            self.subNodes[-1][0].update(toCheck,checkType,add=1)
        
        else:
           
            (node1,node2),checkResult=self.checkNode(rule,checkType)
            
            
            for node in range(node1,node2+1):
                
                # print(checkType)
                
                self.subNodes[node][0].update(toCheck,checkType,add=1)
           
                
        
    def printTree(self,level=1,add=0):
        level=level+add
        print(f'{level} root node: {self.data}')
        if self.subNodes[0][0]!=None:
        
            length2=self.subNodes.shape[0]
            
            for node in range(length2):
                print(f'{level} level: {node} subnode: value: {self.subNodes[node][0].data}, counts: {self.subNodes[node][1]}')
                self.subNodes[node][0].printTree(level,1)
                
                
    def getSubNodes(self,node0):
        
        node0=node0
        
        ruleSet=np.full([1,3],None)
        
        stateType=0
        
        stateTypeRuleSet=np.full([1,3],None)
        
        if stateType==0:
        
            # lamp state
            # nodesDataInt: np.array([[torch:data,int:count]])
            nodesDataInt=np.full([1,2],None)
            nodesDataInt[0,0]=self.subNodes[node0][0].data
            nodesDataInt[0,1]=self.subNodes[node0][1]
            
            ruleSet[0,stateType]=nodesDataInt
            

        
        lengthNode1=self.subNodes[node0][0].subNodes.shape[0]
        
        for node in range(lengthNode1):
            stateType=1
            # ruleSubLevel=np.full([1,lengthNode],None)
            
            nodesDataInt=np.full([1,2],None)
            nodesDataInt[0,0]=self.subNodes[node0][0].subNodes[node][0].data
            nodesDataInt[0,1]=self.subNodes[node0][0].subNodes[node][1]
            ruleSet[0,stateType]=nodesDataInt

            

            lengthNode2=self.subNodes[node0][0].subNodes[node][0].subNodes.shape[0]
            
            for node2 in range(lengthNode2):
                stateType=2
                # ruleSubLevel=np.full([1,lengthNode],None)
                
                nodesDataInt=np.full([1,2],None)
                nodesDataInt[0,0]=self.subNodes[node0][0].subNodes[node][0].subNodes[node2][0].data
                nodesDataInt[0,1]=self.subNodes[node0][0].subNodes[node][0].subNodes[node2][1]
                ruleSet[0,stateType]=nodesDataInt
                
                lengthNode3=self.subNodes[node0][0].subNodes[node][0].subNodes[node2][0].subNodes.shape[0]
                # for node3 in range(lengthNode3):
                #     stateType=3
                #     nodesDataInt=np.full([1,2],None)
                #     nodesDataInt[0,0]=self.subNodes[node0][0].subNodes[node][0].subNodes[node2][0].subNodes[node3][0].data
                #     nodesDataInt[0,1]=self.subNodes[node0][0].subNodes[node][0].subNodes[node2][0].subNodes[node3][1]
                #     ruleSet[0,stateType]=nodesDataInt
                            
                if stateTypeRuleSet[0,0]==None:
                    stateTypeRuleSet=copy.deepcopy(ruleSet)
                else:
                    stateTypeRuleSet=np.concatenate((stateTypeRuleSet,ruleSet),axis=0)
                            
        return stateTypeRuleSet
        # 
    
            
    
    def refineRule(self):
        
        fullSet=np.full([1,1],None)
      
        
        lengthNode0=self.subNodes.shape[0]
        
        if lengthNode0==2:
            # print('yesssssssss')
            pass
        
        for node0 in range(lengthNode0):
            stateType=0
            
            stateTypeRuleSet=self.getSubNodes(node0)
            
            if fullSet[0,0]==None: 
                
                fullSet=copy.deepcopy(stateTypeRuleSet)
            else:
                fullSet=np.concatenate((fullSet,stateTypeRuleSet),axis=0)
        
        
        # fullSet=sortRules(fullSet)
        
        
        fullSetDup=copy.deepcopy(fullSet)
        
        
        # # ################################# for state 3 #######################################
       
        # lenTotal=fullSet.shape[0]
        # toAddArray=np.full([1,4],None)
        # toAddArrays=np.full([1,1],None)
        # toDelteIdx=np.full([1,1],None)

        # notsatisfiedcount=0
        

        # # sort the fullSet data array
        

        # for i in range(0,lenTotal-1):
            
            
        #     fullSet2=np.array([[None]])
            
        #     rule1=fullSet[i]
            
        #     fullSet1=rule1.reshape(-1,4)
            
            
            
        #     if toDelteIdx[0,0]!=None:
                
        #         if i in toDelteIdx:
                    
        #             continue
                
            
        #     for j in range(i+1,lenTotal):

        #         if i!=j:
                    
        #             rule2=fullSet[j]
                    
        #             # %%
        #             if (rule1[0][0,0][0,1]==rule2[0][0,0][0,1] and rule1[1][0,0][0,1]==rule2[1][0,0][0,1]):
        #             # user state  
        #                 if rule1[2][0,0][0,1]==rule2[2][0,0][0,1]:
                            
                            

        #                     fullSet1=np.concatenate((fullSet1, rule2.reshape(-1,4)),axis=0)
                            

        #                     if toDelteIdx[0,0]==None:
        #                         toDelteIdx=np.array([[i]])
        #                         toDelteIdx=np.append(toDelteIdx,np.array([[j]]),axis=1)
                                
        #                     else:
        #                         toDelteIdx=np.append(toDelteIdx,np.array([[i]]),axis=1)
        #                         toDelteIdx=np.append(toDelteIdx,np.array([[j]]),axis=1)
                                
        #     if fullSet1.shape[0]!=1:
            
        #         fullSet2=combineRules(fullSet1)
                
        #         if not(fullSet2[0,0]==None):
        #             fullSetDup=np.concatenate((fullSetDup,fullSet2),axis=0)
                                    
        # if not(toDelteIdx[0,0]==None):
           
        #     fullSetDup=np.delete(fullSetDup,toDelteIdx,axis=0)


        # fullSetDup=sortRules(fullSetDup)

        print(f'fullSetDup shape: {fullSetDup.shape}')
        print(f'fullSetDup : {fullSetDup}')
   
        return fullSetDup

    
# %%