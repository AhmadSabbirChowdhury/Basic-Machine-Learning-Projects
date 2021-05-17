# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 23:48:37 2020

@author: DELL
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X,y)

#predicting a new result
y_pred = regressor.predict([[6.5]])
#The predict method expected 2D array

plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.show()

#Visualizing the result (higher reolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

