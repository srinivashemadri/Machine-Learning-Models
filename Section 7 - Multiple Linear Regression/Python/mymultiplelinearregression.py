import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:, 4 ].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,3]= labelencoder_X.fit_transform(X[:,3])
onehotencoder= OneHotEncoder(categorical_features = [3])
X=onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=0)

#Avoiding dummy variable trap
X=X[:,1:]

#Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test results
y_pred= regressor.predict(X_test)

#Y= c + m1*x1 + m2*x2 + m3*x3 + ………….
# we represent like m0*x0 + m1*x1 + m2*x2 + ....mn*xn, where x0=1(always)
#So we are adding a column of 1's at beginning 
import statsmodels.api as sm
X=np.append(arr= np.ones((50,1)).astype(int), values = X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#2nd attribute has highest p value
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#1st attribute has highest p value
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#remove 4th attriubute
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#remove 5th attribute
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()
