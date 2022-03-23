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
from sklearn.impute import SimpleImputer

import sys

random.seed(0)

# kFold = StratifiedKFold(n_splits=5)

data = pd.read_csv('./data2/processed.cleveland.data.csv', header=None)

# print(data.head())

cols_with_missing = [col for col in data.columns if (data[col]=='?').any()]

data=data.replace('?', 999)

for j in cols_with_missing:

    for i in range(data.values.shape[0]):

        if data[j].values[i]!=np.nan:

            data[j].values[i]=float(data[j].values[i])


data=data.replace(999, np.nan)

fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset = pd.DataFrame(fill_NaN.fit_transform(data))
dataset.columns = data.columns
dataset.index = data.index

dataset.to_csv('./data2/heart_clearned_df.csv', sep='\t')


# print(dataset.head())

# print(data.isnull())

# print(data.info())

# sys.exit()


X = np.array(dataset.values[:,:-1]).astype('float32')

# print(X.shape)



y = np.array(dataset.values[:,-1]).astype('float32')

# print(list(set(y)))

# sys.exit()

np.save('./data2/heart_x.npy',X)
np.save('./data2/heart_y.npy',y)

# sys.exit()

sc = StandardScaler()
X = sc.fit_transform(X)


dump(sc, './data2/heart_std_scaler.bin', compress=True)




# print(y.shape)




class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()

        self.layer1=nn.Linear(input_dim,50)
        self.layer2=nn.Linear(50,50)
        self.layer3=nn.Linear(50,20)
        self.layer4=nn.Linear(20,5)

    def forward(self,x):
        # x = torch.nn.Flatten(x)
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x = F.softmax(self.layer4(x), dim=1)

        return x


model=Model(13)




optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.CrossEntropyLoss()

epochs=100000

batch_size=32

for epoch in range(epochs):

    shuffle=np.random.permutation(X.shape[0])
    X=X[shuffle,:]
    y=y[shuffle]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    len_batches=X_train.shape[0]/batch_size

    X_train=torch.from_numpy(X_train).type(torch.float32)
    Y_train=torch.from_numpy(Y_train).type(torch.LongTensor)

    # print(f'X_train: {X_train.shape}')
    # print(f'Y_train: {Y_train.shape}')

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

        # print(f'output: {output.shape}, y_train: {y_train.shape}')

        loss=loss_fn(output,y_train)

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

           

            for idx_test in range(Y_test.shape[0]):
                if torch.argmax(predicted[[idx_test]], dim=1).item() == Y_test[idx_test]:
                    acc+=1
            
            acc=acc/Y_test.shape[0]

        print(f'epoch:{epoch}, batch: {idx_batch},  loss:{loss},  accuracy:{acc}')





torch.save(model.state_dict(), './data2/heart_nn_classifier.pth')

