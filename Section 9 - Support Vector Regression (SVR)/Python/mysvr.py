import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2 ].values
y = dataset.iloc[:, 2:3].values



#feature scaling ( Most of the algorithms use feauture scaling
#inbuit but svr class is less oftenly used so we need to 
# feauture scale our data manually)
from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
SC_y = StandardScaler()
X= SC_X.fit_transform(X)
y= SC_y.fit_transform(y)

#fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#plotting svr

plt.scatter(X,y)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#since we applied feature scaling to the data, we can't simply give input as 6.5, it will give us a wrong prediction
#regressor.predict(SC_X.transform(arr))  by executing it gives us scaled output, so we need to inverse 
k= SC_X.transform(np.array([[13]]))
y_pred=SC_y.inverse_transform(regressor.predict(k))
