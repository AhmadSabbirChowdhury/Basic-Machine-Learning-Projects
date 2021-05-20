# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:24:49 2020

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing IRIS dataset
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)
#['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']

#What features the iris flowers has
iris.feature_names
...['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']
... 

#Make a dataframe, (iris.data = row and columns = features name)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

#Set the target, means which species of IRIS flower is this
df['target'] = iris.target
df.head()

iris.target_names
#['setosa', 'versicolor', 'virginica'], 0=setosa; 1=versicolor; 2=virginica

#checking which species is from which row in the dataframe
df[df.target==1].head()
df[df.target==2].head()
#0-50 = setosa, 51-100 = versicolor, 101-150 = virginica
#target==1; == versicolor

#creating a new column for flower name
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()

%matplotlib inline

#Creating seperate dataframe for each species
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

df0.head()
df1.head()

#Scatter plot using 'sepal' length and width
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color='red', marker='.')

#Scatter plot using 'petal' length and width
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color='red', marker='.')

#Spliting the data based on Training and Test set
from sklearn.model_selection import train_test_split

#Drop Column
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train) #120 train dataset out of 150 (80%)
len(X_test)     #30 test dataset (20%)
len(y_train)

#Classifier SVC
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

#Checking accuracy score
model.score(X_test, y_test) #Score= 1.0

#Predict the test set results
y_pred = model.predict(X_test)








