# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:33:16 2020

@author: DELL
"""

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

#importing dataset
df = pd.read_csv("income.csv")
df.head()

#scattering a plot based on age and income to see different clusters
plt.scatter(df['Age'], df['Income($)'])

#defyning cluster number
km = KMeans(n_clusters=3)

#prediction
y_predicted = km.fit_predict(df[['Age','Income($)']])

df['cluster'] = y_predicted
df.head()

#plotting differnt clusters
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['Age'], df1['Income($)'], color='green')
plt.scatter(df2['Age'], df2['Income($)'], color='red')
plt.scatter(df3['Age'], df3['Income($)'], color='black')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

#Preprocessing using min max scaler (Scalling the age and income)
scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
df.head()

#plt.scatter(df.Age,df['Income($)'])
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted
df['cluster']=y_predicted
df.head()

#finidng the centroids
km.cluster_centers_

#plotting again after Scalling and putting the centroid
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['Age'], df1['Income($)'], color='green')
plt.scatter(df2['Age'], df2['Income($)'], color='red')
plt.scatter(df3['Age'], df3['Income($)'], color='black')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
            color='purple', marker='+', label='centroid')
            #here taking all the x & y centers of centroid
plt.legend()

#Elbow Plot
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)











