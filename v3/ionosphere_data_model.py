import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

from sklearn.preprocessing import LabelEncoder
import pickle

import random

from joblib import dump, load

import torch


import math

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler

import sys

random.seed(0)

# kFold = StratifiedKFold(n_splits=5)

data = pd.read_csv('./data2/ionosphere.data.csv', header=None)



# print(data.head())

# print(data.info())

# print(data.isnull().sum())

# print(data.iloc[2,:])
# sys.exit()



X = np.array(data.values[1:,0:34]).astype('float32')

X=np.delete(X, np.array([[1]]), axis=1)

print(X.shape)

# sys.exit()

y = np.array(data.values[1:,34])

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y).astype('float32')

# print(y.shape)

np.save('./data2/ionosphere_x.npy',X)
np.save('./data2/ionosphere_y.npy',y)

# sys.exit()



class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        num_input_features=33

       
       
       
        num_output_features=1

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(num_input_features,34),
            torch.nn.ReLU(),
            # torch.nn.Linear(60,60),
            # torch.nn.ReLU(),
            # torch.nn.Linear(60,10),
            # torch.nn.ReLU(),
            torch.nn.Linear(34,num_output_features),
            torch.nn.Sigmoid()

            )

    def forward(self,x):
        x = self.flatten(x)
        y=self.linear_relu_stack(x)

        return y


model=Model()

# model.load_state_dict(torch.load('./data2/iris_nn_classifier.pth'))

optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn=torch.nn.BCELoss()

epochs=10000

batch_size=32

for epoch in range(epochs):

    shuffle=np.random.permutation(X.shape[0])
    X=X[shuffle,:]
    y=y[shuffle]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    len_batches=X_train.shape[0]/batch_size

    X_train=torch.from_numpy(X_train).type(torch.float32)
    Y_train=torch.from_numpy(Y_train)   

    # print(f'X_train: {X_train.shape}')
    # print(f'Y_train: {Y_train.shape}')

    # sys.exit()

    for idx_batch in range(math.ceil(len_batches)):

        x_train=None
        y_train=None


        if idx_batch==math.ceil(len_batches)-1:
            x_train=X_train[idx_batch*batch_size:,:]
            y_train=Y_train[idx_batch*batch_size:]

        else:
            x_train=X_train[idx_batch*batch_size:(idx_batch+1)*batch_size,:]
            y_train=Y_train[idx_batch*batch_size:(idx_batch+1)*batch_size]

        loss=0

        output=model(x_train)

        # print(f'y_train: {y_train.shape}')
        # print(f'output: {output.shape}')

        # sys.exit()

        loss=loss_fn(output,y_train.reshape(-1,1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        acc=0

        with torch.no_grad():

            predicted=model(torch.from_numpy(X_test).type(torch.float32))

            # Y_test=Y_test.astype(np.int)

            # Y_test=torch.from_numpy(Y_test)
            
            # print(predicted)
            # print(torch.argmax(predicted[[0]], dim=1))

           

            Y_test=Y_test.astype(np.float32)
            
            acc=(predicted.reshape(-1).detach().numpy().astype(np.float32).round()==Y_test).mean()

        print(f'epoch:{epoch}, batch: {idx_batch},  loss:{loss},  accuracy:{acc}')





torch.save(model.state_dict(), './data2/inosphere_nn_classifier.pth')

