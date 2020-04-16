from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv(r'C:\Users\Lenovo\Desktop\ML\Mall\Mall_Customers.csv')
a=df.describe()

df.rename(index=str,columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'Score'},inplace=True)

sns.lmplot(x='Income',y='Score',data=df)

x=df.drop(['CustomerID','Gender'],axis=1)
sns.pairplot(df.drop('CustomerID',axis=1),hue='Gender',aspect=1.5)


#Identifying Value of K
from sklearn.cluster import KMeans
clusters=[]
    for i in range(1,11):
        km=KMeans(n_clusters=i).fit(x)
        clusters.append(km.inertia_)
    print(clusters)
    
    
fig,ax=plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)),y=clusters,ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')


ax.annotate('Possible Elbow Point',xy=(3,140000),xytext=(3,50000),xycoords='data',arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='blue',lw=2))
ax.annotate('Possible Elbow Point',xy=(5,80000),xytext=(5,150000),xycoords='data',arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='blue',lw=2))

km3=KMeans(n_clusters=3).fit(x)
x['Labels']=km3.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(x['Income'],x['Score'],hue=x['Labels'],palette=sns.color_palette('hls',3))
plt.title('Kmeans with 3 clusters')


km3=KMeans(n_clusters=5).fit(x)
x['Labels']=km3.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(x['Income'],x['Score'],hue=x['Labels'],palette=sns.color_palette('hls',5))
plt.title('Kmeans with 5 clusters')






