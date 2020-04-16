import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# IMPORT CSV FILES

train_data=pd.read_csv(r'C:\Users\Lenovo\Desktop\ML\Titanic\train.csv')
test_data=pd.read_csv(r'C:\Users\Lenovo\Desktop\ML\Titanic\test.csv')

#CHECK THE DATA SET
train_data.head()

#DESCRIBE THE ENTIRE DATASET'S VALUES FOR MEAN,MODE,MIN,MAX,ETC

train_data.describe()

#NAMES OF COLOUMNS

train_data.columns

#DATATYPE OF EACH COLUMN

train_data.dtypes

#CHECK NO. OF MISSING VALUES
train_data.isnull().sum()

# NO. OF SURVIVED PEOPLE
train_data.Survived.value_counts()

plt =train_data.Survived.value_counts().plot('bar')
plt.set_xlabel('Survived or Not')
plt.set_ylabel('Passenger Count')

#FOR PCLASSS

train_data.Pclass.value_counts().sort_index()

plt=train_data.Pclass.value_counts().sort_index().plot('bar')
plt.set_xlabel("Class")
plt.set_ylabel("Passenger Count")

#Tablular depiction of the above plot

train_data[["Pclass","Survived"]].groupby('Pclass').count()

#TABULAR DEPICTION OF SURVIVED PASSENGER IN DIFF. CLASS

train_data[["Pclass","Survived"]].groupby("Pclass").sum()

#Probabilty of survival

plt=train_data[["Pclass","Survived"]].groupby("Pclass").mean().Survived.plot('bar')
plt.set_xlabel("Pclass")
plt.set_ylabel("Survival Probability")

#PLOT FOR NO. OF MALE AND FEMALE

plt=train_data.Sex.value_counts().sort_index().plot('bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Passenger Count')

train_data[['Sex','Survived']].groupby('Sex').count()

train_data[['Sex','Survived']].groupby('Sex').sum()

#PLOT FOR MALE AND FEMALE SURVIVORS

plt=train_data[['Sex','Survived']].groupby('Sex').mean().Survived.plot('bar', title='')
plt.set_xlabel('Sex')
plt.set_ylabel('Survival probability')

#COUNT FOR PEOPLE EMBARKED

plt=train_data.Embarked.value_counts().plot('bar')
plt.set_ylabel('Count')
plt.set_xlabel('Embarked')


#COUNT FOR SIB and SPOUSES
plt=train_data.SibSp.value_counts().plot('bar')
plt.set_xlabel('No. Of Sibsp')
plt.set_ylabel('Passenger Count')


#TYPES OF CLASS THAT BOARD AT VARIOUS STATIONS i.e PCLASS vs EMBARKED

sns.factorplot("Pclass",col="Embarked",data=train_data, kind='count')

#MALE AND FEMALE ON BOARD FROM VARIOUS STATION i.e.SEX vs EMBARKED

sns.factorplot('Sex', col='Embarked', data=train_data,kind='count')

# ADDING COLUMN FOR FAMILY SIZE

train_data["FamilySize"]=train_data['SibSp']+train_data['Parch']+1

#DROP WASTE COLUMN

train_data=train_data.drop(columns=["Ticket","PassengerId","Cabin"])

#LABEL Encoding
train_data['Sex']=train_data['Sex'].map({'male':0,'female':1})

train_data['Embarked']=train_data['Embarked'].map({'C':0,'Q':1,'S':2})

#Create Title Column
train_data["Title"]=train_data.Name.str.extract('([A-Za-z]+)\.',expand=False)
train_data= train_data.drop(columns='Name')

#Compress title column
train_data.Title.value_counts().plot('bar')
train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')


train_data.Title.value_counts().plot('bar')


#ENCODING TITLE INTO INT64

train_data['Title']=train_data['Title'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Others':4})

#CORRELATION MATRIX

corr_matrix=train_data.corr()
plt.figure(figsize=(9,8))
sns.heatmap(data=corr_matrix,cmap='BrBG',annot=True,linewidths=0.5)

#Handling missing values

train_data.isnull().sum()

train_data['Embarked']=train_data['Embarked'].fillna(2)

train_data.isnull().sum()

#REPLACING AGE MISSING VALUES

age_median_train=train_data.Age.median()
train_data.Age=train_data.Age.fillna(age_median_train)

train_data.isnull().sum()


#####################################
#SHUFFLE THE ROWS FOR RANDOM SELECTION
from sklearn.utils import shuffle
train_data=shuffle(train_data)

#DROP SURVIVED
x_train=train_data.drop(columns='Survived')

y_train=train_data.Survived

y_train.dtypes

y_train=pd.DataFrame({'Survived':y_train.values})

x_train.shape

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #to create validation data set

x_training,x_valid,y_training,y_valid=train_test_split(x_train,y_train,test_size=0.2)

logreg_clf=LogisticRegression()
logreg_clf.fit(x_training,y_training)

prediction=logreg_clf.predict(x_valid)

from sklearn.metrics import accuracy_score

accuracy_score(y_valid,prediction)

from sklearn.metrics import confusion_matrix,classification_report

confusion=confusion_matrix(y_valid,prediction,labels=[1,0])
print(confusion)

report=classification_report(y_valid,prediction)
print(report)



#Test Data Preprocessing 

test_data.head

print(test_data.dtypes)

plt=test_data.Pclass.value_counts().sort_index().plot('bar')
plt.set_xlabel='Pclass'
plt.set_ylabel='Passenger count'

print(test_data.describe())

print(test_data.dtypes)

print(test_data.isnull().sum())

test_data['FamilySize']=test_data['SibSp']+test_data['Parch']+1

test_data=test_data.drop(columns=['Ticket','PassengerId','Cabin'])

test_data['Sex']=test_data['Sex'].map({'male':0,'female':1})
test_data['Embarked']=test_data['Embarked'].map({'C':0,'Q':1,'S':2})

test_data['Title']=test_data.Name.str.extract('([A-Za-z]+)\.',expand=False)
test_data=test_data.drop(['Name'],axis=1)

test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

test_data['Title']=test_data['Title'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Others':4})


test_data['Embarked']=test_data['Embarked'].fillna(2)

age_median_test=test_data.Age.median()
test_data.Age=test_data.Age.fillna(age_median_test)
print('Age median: ',age_median_test)

Fare_median_test=test_data.Fare.median()
test_data.Fare=test_data.Fare.fillna(Fare_median_test)
print('Fare median: ',Fare_median_test)

test_data.Title=test_data.Title.fillna(1)
#test_data=test_data.drop(columns=['SibSp','Parch','FamilySize'])
#finalcheck
print(test_data.isnull().sum())





