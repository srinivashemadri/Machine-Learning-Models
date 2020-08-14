import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#importing datasets
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 1/3, random_state=0)

#fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test results
#we create a vector of all the predicted values
y_pred = regressor.predict(X_test)

#visualizing the training set results
#x-axis user experience, y-axis salary
plt.scatter(X_train,Y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test set results
#x-axis user experience, y-axis salary
plt.scatter(X_test,Y_test, color= 'pink')
plt.plot(X_train, regressor.predict(X_train), color='violet')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


