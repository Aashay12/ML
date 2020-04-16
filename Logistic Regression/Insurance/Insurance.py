import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier



test=pd.read_csv(r'C:\Users\Lenovo\Desktop\CODE\ML\My codes\Logistic Regression\Insurance\carInsurance_test.csv')
train=pd.read_csv(r'C:\Users\Lenovo\Desktop\CODE\ML\My codes\Logistic Regression\Insurance\carInsurance_train.csv')


Des=train.describe()
Head=train.head()

#Using corr() to plot a heatmap
corr_matrix=train.corr()
sns.heatmap(corr_matrix,annot=True)

#Pair plot some omp features

imp_feat=['CarInsurance','Age','Balance','HHInsurance','CarLoan','NoOfContacts','DaysPassed','PrevAttempts']
sns.pairplot(train[imp_feat], size=3)
plt.show()

#Scatterplot for age and carinsurance
plt.scatter(train['Age'],train['CarInsurance'])
plt.xlabel("Age")
plt.ylabel("Car Insurance")
plt.show()

#now for categorical features
cat_feat=train.select_dtypes(include=['object']).columns
plt_feat=cat_feat[(cat_feat!='CallStart')&(cat_feat!='CallEnd')]
for feature in plt_feat:
    plt.figure(figsize=(12,8))
    sns.barplot(feature,'CarInsurance',data=train)
    
#Looking for outliers in balance as shown in pairplot
bal_data=pd.DataFrame(train.Balance)
sns.boxplot(data=bal_data)
#Here we can see that there is an outlier thus removing it.
out=train[train['Balance']>80000]
train=train.drop(train[train.index==1742].index)


#Handling missing data and Combining train and test
all=pd.concat([train,test],keys=('train','test'))
all=all.drop(['CarInsurance','Id'],axis=1)

all.isnull().sum()
all.Job.fillna(all.Job.value_counts().idxmax(),inplace=True)
all.Education.fillna('None',inplace=True)
all.Communication.fillna(all.Communication.value_counts().idxmax(),inplace=True)
all.Outcome.fillna('NoPrev',inplace=True)

all.isnull().sum()

#Feature Engineering
#Create an age group based on age bands
all['AgeBand']=pd.cut(all['Age'],5)
print(all['AgeBand'].value_counts())
all.loc[(all['Age']>=17)&(all['Age']<=34),'AgeBin']=1
all.loc[(all['Age']>=34)&(all['Age']<=49),'AgeBin']=2
all.loc[(all['Age']>=49)&(all['Age']<=65),'AgeBin']=3
all.loc[(all['Age']>=65)&(all['Age']<=80),'AgeBin']=4
all.loc[(all['Age']>=80)&(all['Age']<=95),'AgeBin']=5
all['AgeBin']=all['AgeBin'].astype(int)

#Create Balance Group
all['BalanceBand']=pd.cut(all['Balance'],5)
print(all['BalanceBand'].value_counts())
all.loc[(all['Balance']>=-3150)&(all['Balance']<=8100),'BalanceBin']=1
all.loc[(all['Balance']>=8100)&(all['Balance']<=19200),'BalanceBin']=2
all.loc[(all['Balance']>=19200)&(all['Balance']<=30400),'BalanceBin']=3
all.loc[(all['Balance']>=30400)&(all['Balance']<=41500),'BalanceBin']=4
all.loc[(all['Balance']>=41500)&(all['Balance']<=52600),'BalanceBin']=5
all['BalanceBin']=all['BalanceBin'].astype(int)

all=all.drop(['Age','AgeBand','Balance','BalanceBand'],axis=1)

#Convert everything into numberical values
all['Education']=all['Education'].replace({'None':0,'primary':2,'secondary':2,'tertiary':3})

all.Job.unique()
all['Job']=all['Job'].replace({'management':0, 'blue-collar':1, 'student':2, 'technician':3, 'admin.':4,'services':5, 'self-employed':6, 'retired':7, 'housemaid':8,'entrepreneur':9, 'unemployed':10})

all.Marital.unique()
all['Marital']=all['Marital'].replace({'single':0, 'married':1, 'divorced':2})

all.Communication.unique()
all['Communication']=all['Communication'].replace({'telephone':0,'cellular':1})


all.LastContactMonth.unique()
all['LastContactMonth']=all['LastContactMonth'].replace({'jan':1, 'may':5, 'jun':6, 'mar':3, 'nov':11, 'jul':7, 'aug':8, 'sep':9, 'apr':4,'feb':2, 'oct':10, 'dec':12})

all.Outcome.unique()
all['Outcome']=all['Outcome'].replace({'NoPrev':0, 'failure':1, 'other':2, 'success':3})



#Engineering Call end and Start
all['CallEnd']=pd.to_datetime(all['CallEnd'])
all['CallStart']=pd.to_datetime(all['CallStart'])
all['CallLen']=((all['CallEnd']-all['CallStart'])/np.timedelta64(1,'m')).astype(float)
all['CallBand']=pd.cut(all['CallLen'],5)
print(all['CallBand'].value_counts())


all.loc[(all['CallLen']>=0)&(all['CallLen']<=11),'CallBin']=1
all.loc[(all['CallLen']>=11)&(all['CallLen']<=22),'CallBin']=2
all.loc[(all['CallLen']>=22)&(all['CallLen']<=33),'CallBin']=3
all.loc[(all['CallLen']>=33)&(all['CallLen']<=44),'CallBin']=4
all.loc[(all['CallLen']>=44)&(all['CallLen']<=55),'CallBin']=5
all['CallBin']=all['CallBin'].astype(float)

all=all.drop(['CallLen','CallStart','CallBand','CallEnd'],axis=1)


#Splitting the datasets
train_target=pd.DataFrame(train['CarInsurance'])
idx=pd.IndexSlice
train=all.loc[idx[['train',],:]]
test=all.loc[idx[['test',],:]]

x_train,x_test,y_train,y_test=train_test_split(train,train_target,test_size=0.2,random_state=42)


#Using Logistic Regression
clf=LogisticRegression()
clf.fit(x_train,y_train)
prediction=clf.predict(x_test)
print('Accuracy score:',accuracy_score(y_test,prediction))
score_clf=cross_val_score(clf,train,train_target,cv=10).mean()
print('Cross Validation Score:',score_clf)
confusion=confusion_matrix(y_test,prediction,labels=[1,0])
print('Confusion Matrix:\n',confusion)
y_pred_Log=clf.predict(test)



#Using KNN
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
print('Accuracy:',accuracy_score(y_test,knn.predict(x_test)))
score_knn=cross_val_score(knn,train,train_target,cv=10).mean()
print('Cross Validation Score:',score_knn)
y_pred_knn=knn.predict(test)


#Using Random Forest
rfc=RandomForestClassifier(n_estimators=1000,max_depth=None,min_samples_split=10,class_weight='balanced')
rfc.fit(x_train,y_train)
pred_rfc=rfc.predict(x_test)
print('Accuracy:',accuracy_score(y_test,pred_rfc))
score_rfc=cross_val_score(rfc,train,train_target,cv=10).mean()
print('Cross Validation Score:',score_rfc)
y_pred_rfc=rfc.predict(test)















