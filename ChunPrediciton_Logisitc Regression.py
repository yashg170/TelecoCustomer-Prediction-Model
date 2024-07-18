import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('TelecoCustomer_Churn.csv')

X=data.drop('Churn',axis=1).values
y=data['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

def initialize_parameters(n):
    w=np.zeros((n,1)) # Weight vector w (n*1)
    b=0 # Bias b (1*1)
    return w,b

def sigmoid(z):
   return 1/(1+np.exp(-z))

def compute_cost_and_gradient(X,y,w,b):
    m = X.shape[0]
    z = np.dot(X,w) +b
    A = sigmoid(z)
    cost =1/m * np.sum(y*np.log(A) + (1-y)*np.log(1-A))

    dw =1/m * np.dot(X.T,A-y)
    db =1/m * np.sum(A-y)

    return cost,dw,db

def gradient_descent(X,y,w,b,learning_rate,num_iterations):
    costs=[]

    for i in range(num_iterations):
        cost,dw,db=compute_cost_and_gradient(X,y,w,b)

        w-=learning_rate*dw
        b-=learning_rate*db

        if i%100==0:
           costs.append(cost)
           print('fIteration {i}: {cost}')

    return w,b,costs

def predict(X,w,b):
    z=np.dot(X,w)+b
    A=sigmoid(z)
    return A>=0.5

def confusion_matrix(y_true,y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def accuracy(TP,TN,FP,FN):
    return (TP+TN)/(TP+TN+FP+FN)

def precision(TP,FP):
    return TP/(TP+FP)

def recall(TP,FN):
    return TP/(TP+FN)

def f1_score(precision,recall):
    return 2*(precision*recall)/(precision+recall)

#Train the model
w,b = initialize_parameters(X_train.shape[1])
w,b,costs = gradient_descent(X_train,y_train,w,b,learning_rate=0.01, num_iterations=1000)

#Make predictions
y_train_pred = predict(X_train,w,b)
y_test_pred = predict(X_test,w,b)

#Evaluate the model
TP,TN,FP,FN = confusion_matrix(y_test,y_test_pred)
acc = accuracy(TP,TN,FP,FN)
prec = precision(TP,FP)
rec = recall(TP,FN)
f1 = f1_score(prec,rec)

#Printing metric values

print(f'Confusion Matrix: TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}')
print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
print(f'Recall: {rec}')
print(f'F1 Score: {f1}')