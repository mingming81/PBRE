import torch 

import numpy as np

import pandas as pd

import math

import sys

import copy

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

other_data_name='./v3/data2'

dataset = pd.read_csv(other_data_name+'/Iris.csv')

# print(dataset.head)

# import sys
# sys.exit()
# %%

def calculateEpsilon(checkType,dataset):
    diff=0
    key_val=dataset.keys()
    
    # print(f'checkType: {checkType}')
    values=sorted(dataset[key_val[checkType]].values, key=lambda x: x)
    
    epsilon=0
    
    for i in range(dataset[key_val[checkType]].values.shape[0]-1):
        
        diff+=abs(values[i+1]-values[i])
        epsilon=math.ceil(diff/(dataset[key_val[checkType]].values.shape[0]-1))
        
    return epsilon


def sortRules(fullSet,idx):
    
    # fullSet=fullSet.reshape(-1,4)
    
    # print(f'idx: {idx}')
    # print(f'fullSet[idx]: {fullSet[0,idx]}')
    
    fullSet=sorted(fullSet,key=lambda x: x[idx][0,0][0,1])
    
    fullSet=np.array(fullSet)
    
    # combination
    
    
    
    return fullSet

# %%

def deleteSim(fullSet1):
    
    toDeleteIdx=np.full([1,1],None)
    
    for i in range(0,fullSet1.shape[0]-1):
        
        rule1=fullSet1[i]
        
        for j in range(i+1, fullSet1.shape[0]):
            
            rule2=fullSet1[j]
            
            num_sim=0
            
            for idx_s in range(1,5):
                
                if rule1[idx_s][0,0][0,1]==rule2[idx_s][0,0][0,1] and rule1[idx_s][0,0][0,2]==rule2[idx_s][0,0][0,2]:
                    num_sim+=1
                    
            if num_sim==4:
                
                rule1[idx_s][0,1]=max(rule1[idx_s][0,1], rule2[idx_s][0,1])
                
                if toDeleteIdx[0,0]==None:
                    toDeleteIdx=np.array([[j]])
                else:
                    toDeleteIdx=np.concatenate((toDeleteIdx, np.array([[j]])),axis=1)
                    
    if not(toDeleteIdx[0,0]==None):
                    
        fullSet1=np.delete(fullSet1, toDeleteIdx,axis=0)
    
    return fullSet1
                
                
            
            

# %%

def combineRules(fullSet1,idx):
    
    fullSet1=sortRules(fullSet1,idx)
    
    combined=fullSet1[0]

    fullSet2=np.array([[None]])
        
    toAddIdx=np.full([1,1],None)
    
    for i in range(1,fullSet1.shape[0]):
        
        toCombined=fullSet1[i]
        
        if toCombined[idx][0,0][0,1]<=combined[idx][0,0][0,2]:

            combined[idx][0,0][0,0]=(toCombined[idx][0,1]*toCombined[idx][0,0][0,0]+combined[idx][0,1]*combined[idx][0,0][0,0])/(toCombined[idx][0,1]+combined[idx][0,1])
            combined[idx][0,0][0,2]=max(toCombined[idx][0,0][0,2],combined[idx][0,0][0,2])
            combined[idx][0,1]=toCombined[idx][0,1]+combined[idx][0,1]
            
            if toAddIdx[0,0]==None:
                toAddIdx=np.array([[i]])
            else:
                toAddIdx=np.concatenate((toAddIdx, np.array([[i]])),axis=1)
                
            
        else:
            if fullSet2[0,0]==None:
                
                fullSet2=copy.deepcopy(combined.reshape(-1,5))
                if toAddIdx[0,0]!=None:
                    for toi in toAddIdx[0]:
                        toAdd=fullSet1[toi]
                        toAdd[idx]=combined[idx]
                        fullSet2=np.concatenate((fullSet2,toAdd.reshape(-1,5)),axis=0)
                
            else:
                
                fullSet2=np.concatenate((fullSet2,combined.reshape(-1,5)),axis=0)
                if toAddIdx[0,0]!=None:
                    for toi in toAddIdx[0]:
                        toAdd=fullSet1[toi]
                        toAdd[idx]=combined[idx]
                        fullSet2=np.concatenate((fullSet2,toAdd.reshape(-1,5)),axis=0)
            
            combined=fullSet1[i]
            toAddIdx=np.full([1,1],None)
      
                
    if fullSet2[0,0]==None:
                
        fullSet2=copy.deepcopy(combined.reshape(-1,5))
        
        if toAddIdx[0,0]!=None:
            for toi in toAddIdx[0]:
                toAdd=fullSet1[toi]
                toAdd[idx]=combined[idx]
                fullSet2=np.concatenate((fullSet2,toAdd.reshape(-1,5)),axis=0)
        
    else:
        
        fullSet2=np.concatenate((fullSet2,combined.reshape(-1,5)),axis=0)
        
        if toAddIdx[0,0]!=None:
            for toi in toAddIdx[0]:
                toAdd=fullSet1[toi]
                toAdd[idx]=combined[idx]
                fullSet2=np.concatenate((fullSet2,toAdd.reshape(-1,5)),axis=0)
                
    fullSet2=deleteSim(fullSet2)
                
    return fullSet2



def combineIdxRules(fullSetx,idx):


    fullSetDupx=copy.deepcopy(fullSetx)
        
        
    # # ################################# for state 3 #######################################
   
    lenTotal=fullSetx.shape[0]

    toDelteIdx=np.full([1,1],None)

    notsatisfiedcount=0
    

    # sort the fullSet data array
    

    for i in range(0,lenTotal-1):
        
        
        fullSet2=np.array([[None]])
        
        rule1=fullSetx[i]
        
        fullSet1=rule1.reshape(-1,5)
        
        
        
        if toDelteIdx[0,0]!=None:
            
            if i in toDelteIdx:
                
                continue
            
        
        for j in range(i+1,lenTotal):

            if i!=j:
                
                rule2=fullSetx[j]
                
                # %%
                if rule1[0][0,0][0,1]==rule2[0][0,0][0,1]:
                # user state  
                    
                    fullSet1=np.concatenate((fullSet1, rule2.reshape(-1,5)),axis=0)
                    

                    if toDelteIdx[0,0]==None:
                        toDelteIdx=np.array([[i]])
                        toDelteIdx=np.append(toDelteIdx,np.array([[j]]),axis=1)
                        
                    else:
                        toDelteIdx=np.append(toDelteIdx,np.array([[i]]),axis=1)
                        toDelteIdx=np.append(toDelteIdx,np.array([[j]]),axis=1)
                            
        if fullSet1.shape[0]!=1:
        
            fullSet2=combineRules(fullSet1,idx)
            
            if not(fullSet2[0,0]==None):
                fullSetDupx=np.concatenate((fullSetDupx,fullSet2),axis=0)
                                
    if not(toDelteIdx[0,0]==None):
       
        fullSetDupx=np.delete(fullSetDupx,toDelteIdx,axis=0)

    return fullSetDupx
   


def nodeRange(nearNodes,rule):
    resultRange=np.full([1,5],None)
    
    lenNearNode=nearNodes.shape[0]
    for i in range(lenNearNode):
        
        resultRangeInt=np.full([1,5],None)
        
        if nearNodes[i][1]>=rule[0,1] and nearNodes[i][2]>=rule[0,2]:
            resultRangeInt[0,0]=(nearNodes[i][0]*nearNodes[i][4]+rule[0,0])/(nearNodes[i][0]+1)
            resultRangeInt[0,1]=rule[0,1]
            resultRangeInt[0,2]=nearNodes[i][2]
            resultRangeInt[0,3]=nearNodes[i][0]+1
            resultRangeInt[0,4]=nearNodes[i][3]
            
        elif nearNodes[i][1]<=rule[0,1] and nearNodes[i][2]>=rule[0,2]:
            resultRangeInt[0,0]=(nearNodes[i][0]*nearNodes[i][4]+rule[0,0])/(nearNodes[i][0]+1)
            resultRangeInt[0,1]=nearNodes[i][1]
            resultRangeInt[0,2]=nearNodes[i][2]
            resultRangeInt[0,3]=nearNodes[i][0]+1
            resultRangeInt[0,4]=nearNodes[i][3]
        
        elif nearNodes[i][1]<=rule[0,1] and nearNodes[i][2]<=rule[0,2]:
            resultRangeInt[0,0]=(nearNodes[i][0]*nearNodes[i][4]+rule[0,0])/(nearNodes[i][0]+1)
            resultRangeInt[0,1]=nearNodes[i][1]
            resultRangeInt[0,2]=rule[0,2]
            resultRangeInt[0,3]=nearNodes[i][0]+1
            resultRangeInt[0,4]=nearNodes[i][3]
                        
        if resultRange[0,0]==None:
            resultRange=copy.deepcopy(resultRangeInt)
        else:
            resultRange=np.concatenate((resultRange,resultRangeInt),axis=0)
    
    return resultRange

# rule1, rule2: np.array([[np.array([["lr",tensor:min,tensor:max]]),int:count]])  (1*2), and each line: 1*5

   
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
            if checkType==0:
                
                if self.subNodes[node][0].data[0,1]==rule[0,1] or self.subNodes[node][0].data[0,2]==rule[0,2]:
                    case=0
                    # return index of subnodes, category of the same state (lamp state), the truth of whether having the same states
                    
                    self.subNodes[node][0].data[0,0]=(self.subNodes[node][1]*self.subNodes[node][0].data[0,0]+rule[0,0])/(self.subNodes[node][1]+1)
                    self.subNodes[node][1]+=1
                    
                    # return the index
                    return (node,node),True
        
        # int:count,tensor:min,tensor:max
        nearNodes=np.full([1,5],None)
        toDeleteNodes=np.full([1,1],None)

        min_dis=0

        if checkType!=0:

            importAtt=np.array([0,1,2,3,4])

            min_dis=calculateEpsilon(checkType, dataset)

            if checkType not in importAtt:

                min_dis=calculateEpsilon(checkType, dataset)


        for node in range(length):
            # ensure that the compared states only support continuous states
            if checkType!=0:
                # this case is responsible for changing the value too, patience equal 
                if (abs(self.subNodes[node][0].data[0,1]-rule[0,1])<=torch.tensor([[min_dis]]).type(torch.float32) or abs(self.subNodes[node][0].data[0,2]-rule[0,2])<=torch.tensor([[min_dis]])).type(torch.float32) or (rule[0,1]>=self.subNodes[node][0].data[0,1] and rule[0,1]<=self.subNodes[node][0].data[0,2]):
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
                        nearNodes[0][4]=self.subNodes[node][0].data[0,0] #torch:abg
                        
                    else:
                        nearNodeInter=np.full([1,4],None)
                        nearNodeInter[0][0]=self.subNodes[node][1] # int: count
                        nearNodeInter[0][1]=self.subNodes[node][0].data[0,1] #torch:min
                        nearNodeInter[0][2]=self.subNodes[node][0].data[0,2] #torch:max
                        nearNodeInter[0][3]=self.subNodes[node][0].subNodes #torch: subNodes
                        nearNodeInter[0][4]=self.subNodes[node][0].data[0,0] #torch:avg
                        
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
        
        if checkType==5:
            return
        
       
      
        rule=toCheck[0,checkType]
        
        rule1=rule[0,1].reshape((1,-1))
        
        rule=np.concatenate((rule,rule1),axis=1)
        # print(f'rule[0,2]: {rule[0,2]}')

        rule[0,0]=rule[0,0].type(torch.float32)
        rule[0,1]=rule[0,1].type(torch.float32)
        rule[0,2]=torch.tensor([[rule[0,2]]]).reshape(1,-1).type(torch.float32)

   
        
        if self.subNodes[-1][0]==None:
            self.insert(rule,-1)
            self.subNodes[-1][1]=1
            self.subNodes[-1][0].update(toCheck,checkType,add=1)
        
        else:
           
            (node1,node2),checkResult=self.checkNode(rule,checkType)
            
            
            for node in range(node1,node2+1):
   
                self.subNodes[node][0].update(toCheck,checkType,add=1)


    def pruneNodeCell(self, checkType=1, add=0):

        checkType=checkType+add

        min_dis=0
        
        if checkType==5:
            return

        if checkType==0:
            return
   

        importAtt=np.array([0,1,2,3,4])

        min_dis=calculateEpsilon(checkType, dataset)

        if checkType not in importAtt:

            min_dis=calculateEpsilon(checkType, dataset)


        self.subNodes=np.array(sorted(self.subNodes, key=lambda x: x[0].data[0,1]))


        lengthNode=None

        lengthNode=self.subNodes.shape[0]

        toDeleteIdx=np.array([[None]])

        combineSet=np.full([1,2],None)

        if lengthNode!=None and lengthNode>1:

            combined=copy.deepcopy(self.subNodes[0])
            toDeleteIdx[0,0]=0

            for i in range(1, lengthNode):

                if abs(combined[0].data[0,1]-self.subNodes[i][0].data[0,1])<torch.tensor([[min_dis]]).type(torch.float32) or abs(combined[0].data[0,2]-self.subNodes[i][0].data[0,1])<torch.tensor([[min_dis]]).type(torch.float32) or abs(combined[0].data[0,1]-self.subNodes[i][0].data[0,2])<torch.tensor([[min_dis]]).type(torch.float32) or abs(combined[0].data[0,2]-self.subNodes[i][0].data[0,2])<torch.tensor([[min_dis]]).type(torch.float32): 

                    combined[0].data[0,0]=(combined[1]*combined[0].data[0,0]+self.subNodes[i][1]*self.subNodes[i][0].data[0,0])/(combined[1]+self.subNodes[i][1])
                    combined[0].data[0,2]=copy.deepcopy(max(combined[0].data[0,2], self.subNodes[i][0].data[0,2]))
                    combined[0].subNodes=np.concatenate((combined[0].subNodes,self.subNodes[i][0].subNodes),axis=0)
                    combined[1]=combined[1]+self.subNodes[i][1]

                    if toDeleteIdx[0,0]==None:
                        toDeleteIdx[0,0]=i
                    else:
                        toDeleteIdx=np.concatenate((toDeleteIdx, np.array([[i]])),axis=1)


                else:

                    if combineSet[0,0]==None:
                        combineSet=copy.deepcopy(combined).reshape(1,2)
                    else:
                        combineSet=np.concatenate((combineSet,combined.reshape(1,2)),axis=0)

                    if toDeleteIdx[0,0]==None:
                        toDeleteIdx[0,0]=i
                    else:
                        toDeleteIdx=np.concatenate((toDeleteIdx, np.array([[i]])),axis=1)

                    combined=copy.deepcopy(self.subNodes[i])


            if combineSet[0,0]==None:
                combineSet=copy.deepcopy(combined).reshape(1,2)
            else:
                combineSet=np.concatenate((combineSet,combined.reshape(1,2)),axis=0)

            if toDeleteIdx[0,0]==None:
                toDeleteIdx[0,0]=i
            else:
                toDeleteIdx=np.concatenate((toDeleteIdx, np.array([[i]])),axis=1)




            self.subNodes=np.concatenate((self.subNodes, combineSet),axis=0)

            self.subNodes=np.delete(self.subNodes,toDeleteIdx.astype(np.int),axis=0)

        lengthNode2=self.subNodes.shape[0]

        for j in range(lengthNode2):

            self.subNodes[j][0].pruneNodeCell(checkType, add=1)


    def pruneNodes(self):
        
        lenclass=self.subNodes.shape[0]
        
        for k in range(lenclass):
            
            # if k==1:
            #     print('yes')

            self.subNodes[k][0].pruneNodeCell()
           
                
        
                
    def getSubNodes(self,stateTypeRuleSet2, ruleSet, stateType=0, add=0):
        
        stateType=stateType+add
        
        # if stateTypeRuleSet[0,0]!=None:        
        #     stateTypeRuleSet2=copy.deepcopy(stateTypeRuleSet)
        
        if stateType==5:
            
            return 
        
        lengthNode=self.subNodes.shape[0]        
        
        for i in range(lengthNode):
            nodesDataInt=np.full([1,2],None)
            nodesDataInt[0,0]=self.subNodes[i][0].data
            nodesDataInt[0,1]=self.subNodes[i][1]
            
            ruleSet[0,stateType]=nodesDataInt
            
            if stateType==4:
                # if stateTypeRuleSet[0,0]==None:
                    # stateTypeRuleSet=copy.deepcopy(ruleSet)
                stateTypeRuleSet2.append(copy.deepcopy(ruleSet))
                # else:
                #     stateTypeRuleSet=np.concatenate((stateTypeRuleSet,ruleSet),axis=0)
                    
            
          
            self.subNodes[i][0].getSubNodes(stateTypeRuleSet2,ruleSet, stateType, add=1)
            
        # else:
        #     return stateTypeRuleSet
    
    
    def refineRule(self, checkType=0, add=0):
        
        fullSet=np.full([1,1],None)
      
        
        lengthNode0=self.subNodes.shape[0]
        
        stateTypeRuleSet2=[]
        ruleSet=np.full([1,5],None)
        
        self.getSubNodes(stateTypeRuleSet2, ruleSet)
        
        if fullSet[0,0]==None:
            fullSet=copy.deepcopy(stateTypeRuleSet2[0])
            lenR=len(stateTypeRuleSet2)
            for i in range(1,lenR):
                fullSet=np.concatenate((fullSet, stateTypeRuleSet2[i]),axis=0)
            
        else:
            lenR=len(stateTypeRuleSet2)
            for i in range(0,lenR):
                fullSet=np.concatenate((fullSet, stateTypeRuleSet2[i]),axis=0)
        
        # print(f'fullSet shape: {fullSet.shape}')
        
       
        # ################################# for state 3 #######################################
        
        for i in range(5):
            
            print(f'combine: {i}')
        
            fullSet=combineIdxRules(fullSet, i)
          
        for i in range(5):
            
            print(f'sort: {i}')
            # if i==187:
            #     print('yes')
            fullSet=sortRules(fullSet,4-i)
           

      
   
        return fullSet
        return fullSetDup
        
        

