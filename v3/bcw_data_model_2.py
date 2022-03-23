import numpy as np
import pandas as pd
import seaborn as sns
import warnings

import torch

import math

from joblib import dump, load

from sklearn.preprocessing import StandardScaler

import random
import numpy as np 
import pandas as pd 

import torch.optim as optim

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


warnings.simplefilter(action='ignore', category=Warning)

random.seed(0)


sc=load('./data2/bcw_std_scaler.bin')

dataset = pd.read_csv('./data2/bcw_data.csv')

dataset = dataset.drop('Unnamed: 32', axis =1)



X = dataset.iloc[:,2:].values

y = dataset.iloc[:, 1:2].values


X=sc.transform(X)


le = LabelEncoder()
y = le.fit_transform(y.ravel())


# X=torch.from_numpy(X).type(torch.float32)
# y=torch.from_numpy(y).type(torch.float32)


accuracy_scores = {}





# https://towardsdatascience.com/deep-learning-in-winonsin-breast-cancer-diagnosis-6bab13838abd#:~:text=standard%20error%2C%20etc.)%20and%20experimented%20three%20types%20of%20deep%20neuron%20networks%3A%201%2C%202%2C%20and%203%20hidden%20layers%20of%2030%20neurons

class ServiceModel(torch.nn.Module):
    def __init__(self):
        super(ServiceModel,self).__init__()
        
        num_input_features=30

       
        num_hidden=30 
       
        num_output_features=1

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(

            torch.nn.Linear(num_input_features,num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden,30),
            torch.nn.ReLU(),
            torch.nn.Linear(30,30),
            torch.nn.ReLU(),
            torch.nn.Linear(30,num_output_features),
            torch.nn.Sigmoid()

            )

      
    def forward(self,x):
        x = self.flatten(x)
        y=self.linear_relu_stack(x)

        return y

# from sklearn.svm import SVC
# classifier = SVC(C=1, gamma=0.8,kernel='linear', random_state=0)

# # predictor('svm', {'C': 1, 'gamma': 0.8,
# #           'kernel': 'linear', 'random_state': 0})

# classifier.fit(X_train, y_train)

classifier=ServiceModel()

criterion=torch.nn.BCELoss()
optimizer=optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)


epochs=10000

batch_size=32



for epoch in range(epochs):


    shuffle=np.random.permutation(X.shape[0])

    X=X[shuffle,:]
    y=y[shuffle]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    
    

    len_batches=X_train.shape[0]/batch_size

    X_train=torch.from_numpy(X_train).type(torch.float32)
    Y_train=torch.from_numpy(Y_train).type(torch.float32)

    for idx_batch in range(math.ceil(len_batches)):

        # print(f'epoch: {epoch}')

        # print(f'idx_batch: {idx_batch}')

        x_train=None
        y_train=None


        if idx_batch==math.ceil(len_batches)-1:
            x_train=X_train[idx_batch*batch_size:,:]
            y_train=Y_train[idx_batch*batch_size:]

        else:
            x_train=X_train[idx_batch*batch_size:(idx_batch+1)*batch_size,:]
            y_train=Y_train[idx_batch*batch_size:(idx_batch+1)*batch_size]

        loss=0

        

        # for i in range(x_train.shape[0]):

        output=classifier(x_train)

        # print(f'output shape : {output.shape}')
        # print(f'y_train shape : {Y_train.shape}')

        print(f'')

        loss=criterion(output,y_train.reshape(-1,1))

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        acc=0

        with torch.no_grad():

            predicted=classifier(torch.from_numpy(X_test).type(torch.float32))
            Y_test=Y_test.astype(np.float32)
            acc=(predicted.reshape(-1).detach().numpy().astype(np.float32).round()==Y_test).mean()


        # if epoch%50==0:

        print(f'epoch:{epoch}, batch: {idx_batch},  loss:{loss},  accuracy:{acc}')



        



torch.save(classifier.state_dict(), './data2/bcw_nn_classifier.pth')





# print('''Making Confusion Matrix''')

# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# y_pred = classifier(X_test)
# cm = confusion_matrix(y_test, y_pred)
# print(cm,'\n')
# print('True Positives :',cm[0][0])
# print('False Positives :',cm[0][1])
# print('False Negatives :',cm[1][0])
# print('True Negatives :', cm[0][1],'\n')


# print('''Classification Report''')
# print(classification_report(y_test, y_pred,
#       target_names=['M', 'B'], zero_division=1))

# print('''Evaluating Model Performance''')
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy,'\n')

# print('''Applying K-Fold Cross validation''')
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# accuracy_scores[classifier] = accuracies.mean()*100
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100),'\n')  






