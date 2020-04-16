# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:50:30 2020

@author: Aashay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

test_data=pd.read_csv(r'C:\Users\Lenovo\Desktop\CODE\ML\My codes\Loan\test.csv')
train_data=pd.read_csv(r'C:\Users\Lenovo\Desktop\CODE\ML\My codes\Loan\train.csv')

train_data.head()

describe=train_data.describe()

train_data.info()

train_data.isnull().sum()

train_data.Gender.value_counts(normalize=True).plot('bar')

train_data.groupby(["Gender", "Education"])["Loan_ID"].count().plot('bar')

#Handling missing values in Gender

gen=train_data.Gender.value_counts().idxmax()
gen1=test_data.Gender.value_counts().idxmax()
train_data.Gender.fillna(gen, inplace=True)
test_data.Gender.fillna(gen1, inplace=True)


# Handling missing values in married
train_data.Married.fillna(train_data.Married.value_counts().idxmax(),inplace=True)
test_data.Married.fillna(test_data.Married.value_counts().idxmax(),inplace=True)


#Handling missing values in dependents
train_data.Dependents.fillna(train_data.Dependents.value_counts().idxmax(),inplace=True)
test_data.Dependents.fillna(test_data.Dependents.value_counts().idxmax(),inplace=True)


#Handling missing values in Self_Employed
train_data.Self_Employed.fillna(train_data.Self_Employed.value_counts().idxmax(),inplace=True)
test_data.Self_Employed.fillna(test_data.Self_Employed.value_counts().idxmax(),inplace=True)


#Handling missing values in LoanAmount
train_data.LoanAmount.fillna(train_data.LoanAmount.value_counts().mean(),inplace=True)
test_data.LoanAmount.fillna(test_data.LoanAmount.value_counts().mean(),inplace=True)


#Handling missing values in Loan_Amount_Term
train_data.Loan_Amount_Term.fillna(train_data.Loan_Amount_Term.value_counts().idxmax(),inplace=True)
test_data.Loan_Amount_Term.fillna(test_data.Loan_Amount_Term.value_counts().idxmax(),inplace=True)



#Handling missing values in Credit_History
train_data.Credit_History.fillna(train_data.Credit_History.value_counts().idxmax(),inplace=True)
test_data.Credit_History.fillna(test_data.Credit_History.value_counts().idxmax(),inplace=True)

train_data.isnull().sum()
test_data.isnull().sum()


train_data.drop(['Loan_ID'],axis=1,inplace=True)
test_data.drop(['Loan_ID'],axis=1,inplace=True)

#Encoding all values inthe dataset

train_data['Gender']=train_data['Gender'].replace({'Male':0,'Female':1})
train_data['Married']=train_data['Married'].replace({'No':0,'Yes':1})
train_data['Education']=train_data['Education'].replace({'Not Graduate':0,'Graduate':1})
train_data['Dependents']=train_data['Dependents'].replace({'3+':3})
train_data['Self_Employed']=train_data['Self_Employed'].replace({'No':0,'Yes':1})
train_data['Property_Area']=train_data['Property_Area'].replace({'Rural':1,'Urban':2,'Semiurban':3})
train_data['Loan_Status']=train_data['Loan_Status'].replace({'N':0,'Y':1})

test_data['Gender']=test_data['Gender'].replace({'Male':0,'Female':1})
test_data['Married']=test_data['Married'].replace({'No':0,'Yes':1})
test_data['Education']=test_data['Education'].replace({'Not Graduate':0,'Graduate':1})
test_data['Dependents']=test_data['Dependents'].replace({'3+':3})
test_data['Self_Employed']=test_data['Self_Employed'].replace({'No':0,'Yes':1})
test_data['Property_Area']=test_data['Property_Area'].replace({'Rural':1,'Urban':2,'Semiurban':3})

#train_data.groupby(['Gender','Education'])['Loan_Status'].count().plot('bar')

X_train=train_data.drop(columns='Loan_Status',axis=1)
Y_train=train_data.Loan_Status
type(y_train)
Y_train=pd.DataFrame(y_train)


#Splitting datasets
x_train,x_valid,y_train,y_valid=train_test_split(X_train,Y_train,test_size=0.2)
logreg_clf=LogisticRegression()
logreg_clf.fit(x_train,y_train)

#Prediciting y_valid
prediction=logreg_clf.predict(x_valid)

#Checking with y_valid
print('Accuracy score:',accuracy_score(y_valid,prediction))

#Confusion Matrix
confusion=confusion_matrix(y_valid,prediction, labels=[1,0])
print("Confusion matrix:\n",confusion)

#Classification Report
report=classification_report(y_valid,prediction)
print("Classification Report:\n",report)

#Apply algorithm on Test Data
prediction_test=logreg_clf.predict(test_data)
test_data['Loan Status']=prediction_test
