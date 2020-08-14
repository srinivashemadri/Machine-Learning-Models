import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2 ]
y = dataset.iloc[:, 2]

from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor(n_estimators=10 , random_state=0)
regressor.fit(X,y)

X1= dataset.iloc[:, 1 ]
X_grid = np.arange(min(X1),max(X1),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

regressor.predict(np.array([[6.5]]))