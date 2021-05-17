# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:34:04 2020

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary.csv')

#Devide the dataset into x and y
X=dataset.iloc[:,:-1].values #means all rows and all columns going down in Y axis
y=dataset.iloc[:,1].values #.values will take all the original data types. (All rows, first column)

#Spliting the data based on Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)
#here in train_test_split, we choose all arrays, test size (10)


#Implement classifier based on SLR
from sklearn.linear_model import LinearRegression
simpleLinearRegrssion = LinearRegression()
simpleLinearRegrssion.fit(X_train,y_train)

#Predict value for those 10 test data set
y_predict = simpleLinearRegrssion.predict(X_test)

#Predicting salary for an specific year which is not in the DATASET


#implement the GRAPH
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, simpleLinearRegrssion.predict(X_train))
plt.show()