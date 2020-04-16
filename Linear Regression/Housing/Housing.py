# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:20:35 2020

@author: Aashay
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score
x=pd.read_csv(r'C:\Users\Lenovo\Desktop\CODE\ML\My codes\Linear Regression\Housing\HousingData.csv')

#Data description
Des=x.describe()

 #correlarion martix 
corr_matrix=x.corr()
sns.heatmap(data=corr_matrix, annot=True)


x.isnull().sum()

#Handling missing values
crim_data=pd.DataFrame(x.CRIM)
sns.boxplot(data=crim_data)
#Since CRIM has outliers we use median to fill missing values
crim_med=x.CRIM.median()
x.CRIM=x.CRIM.fillna(crim_med)

#Handling ZN values
x.ZN=x.ZN.fillna(x.ZN.median())

#Handling INDUS values
indus_data=pd.DataFrame(x.INDUS)
sns.boxplot(data=indus_data)
x.INDUS=x.INDUS.fillna(x.INDUS.mean())

#Handling AGE values
AGE_data=pd.DataFrame(x.AGE)
sns.boxplot(data=AGE_data)
x.AGE=x.AGE.fillna(x.AGE.mean())

#Handling LSTAT values
LSTAT_data=pd.DataFrame(x.LSTAT)
sns.boxplot(data=LSTAT_data)
x.LSTAT=x.LSTAT.fillna(x.LSTAT.median())

#Handling CHAS values
x.CHAS=x.CHAS.fillna(x.CHAS.value_counts().idxmax())

x.isnull().sum()


##Outlier Treament using scatterplot
X=x
'''
fig,ax=plt.subplots(figsize=(16,8))
ax.scatter(data_copy.INDUS,data_copy.TAX)
ax.set_xlabel("proportion of non-retail business acres per town.")
ax.set_ylabel("full-value property-tax rate per \$10,000.")
plt.show()
'''
##Outlier Treament using IQR
'''
q1=data_copy.quantile(0.25)
q3=data_copy.quantile(0.75)
iqr=q3-q1
print(iqr)
bool=(data_copy < (q1 - 1.5 * iqr)) |(data_copy > (q3 + 1.5 * iqr))
'''



##Outlier Treament using Z-score

z=np.abs(stats.zscore(X))
z_score=z
threshold=2.7
print(np.where(z>))
#print(z[55][1])
X.shape
X=X[(z<2.7).all(axis=1)]
X.shape



y=X.MEDV
X=X.drop(["MEDV"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
x_train.isnull().sum()
Lin=LinearRegression()
Lin.fit(x_train,y_train)
print(Lin)
y_pred=Lin.predict(x_test)
prediction=y_pred

print("Coefficient:\n",Lin.coef_)
print("Mean Squared Error: %.2f" % mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


