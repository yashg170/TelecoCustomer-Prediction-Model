import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# %matplotlib inline

df =pd.read_csv("TelecoCustomer_Churn.csv")
df.sample(5)
# customer ID is useless so drop it

df.drop('customerID',axis=1,inplace=True)
df.dtypes

df.TotalCharges.values

df.MonthlyCharges.values

pd.to_numeric(df.TotalCharges)

pd.to_numeric(df.TotalCharges,errors='coerce').isnull()

df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]

df1=df[df.TotalCharges!=' ']
df1.shape

df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

df1.TotalCharges.dtypes

tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
plt.hist([tenure_churn_yes,tenure_churn_no],color=['red','green'],label=['Churn=Yes','Churn=No'])
plt.legend()

monthlycharges_churn_no=df1[df1.Churn=='No'].MonthlyCharges
monthlycharges_churn_yes=df1[df1.Churn=='Yes'].MonthlyCharges

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([monthlycharges_churn_yes,monthlycharges_churn_no],color=['red','green'],label=['Churn=Yes','Churn=No'])

def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f' {column} : {df[column].unique()}')

print_unique_col_values(df1)

df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)

print_unique_col_values(df1)

yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No' : 0}, inplace=True)

for col in df1:
    print(f'{col}: {df1[col].unique()}')

"""# New Section"""

df1['gender'].replace({'Female':1,'Male':0},inplace=True)

df1['gender'].unique()

df2 = pd.get_dummies(df1,columns=['InternetService','Contract','PaymentMethod'])
df2.columns

dummy_columns = df2.columns[df2.columns.str.startswith(('InternetService_', 'Contract_', 'PaymentMethod_'))]
df2[dummy_columns] = df2[dummy_columns].astype('uint8')
df2.head()

df2.dtypes

cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
df2.head()

X=df2.drop('Churn',axis=1);
y=df2['Churn']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)

X_train.shape

X_test.shape

X_train[:10]

import tensorflow as tf
from tensorflow import keras

model=keras.Sequential([
    keras.layers.Dense(26,input_shape=(26,),activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)

model.evaluate(X_test,y_test)

yp=model.predict(X_test)
yp[:5]

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred[:5]

from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))

import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""# New Section"""