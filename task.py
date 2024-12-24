import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('Task.csv')
print(dataset)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X-train = ",X_train)
print("X-test = ",X_test)
print("y-train = ",y_train)
print("y_test = ",y_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))
from sklearn.metrics import r2_score
print(r2_score(y_train, regressor.predict(X_train)))
print(r2_score(y_test, y_pred))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, regressor.predict(X_train)))
print(mean_squared_error(y_test, y_pred))
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X-train = ",X_train)
print("X-test = ",X_test)
print("y-train = ",y_train)
print("y_test = ",y_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))
from sklearn.metrics import r2_score
print(r2_score(y_train, regressor.predict(X_train)))
print(r2_score(y_test, y_pred))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, regressor.predict(X_train)))
print(mean_squared_error(y_test, y_pred))
