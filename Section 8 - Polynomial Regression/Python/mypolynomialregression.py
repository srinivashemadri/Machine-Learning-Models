import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2 ]
y = dataset.iloc[:, 2]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
pol_reg=PolynomialFeatures( degree = 4)
X_poly = pol_reg.fit_transform(X)
lin_reg.fit(X_poly,y)
pol_reg.fit(X_poly,y)

plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X_poly), color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

arr = np.arange(1, dtype = float).reshape(-1,1)
arr[0]=6.5
lin_reg.predict(pol_reg.fit_transform(arr))