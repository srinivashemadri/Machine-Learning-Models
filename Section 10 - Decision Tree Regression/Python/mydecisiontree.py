import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2 ].values
y = dataset.iloc[:, 2:3].values




from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
SC_y = StandardScaler()
X= SC_X.fit_transform(X)
y= SC_y.fit_transform(y)

#fitting decisiontree to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#plotting svr

plt.scatter(X,y , color='blue')
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#since we applied feature scaling to the data, we can't simply give input as 6.5, it will give us a wrong prediction
#regressor.predict(SC_X.transform(arr))  by executing it gives us scaled output, so we need to inverse 
k= np.array([[6.7]])
y_pred=regressor.predict(k)
