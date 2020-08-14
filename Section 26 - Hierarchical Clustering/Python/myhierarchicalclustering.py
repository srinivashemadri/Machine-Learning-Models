import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

dataset= pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dedrogram = sch.dendrogram(sch.linkage(X,method='ward'))
#ward is a method which minimizes the variance within each cluster
#it's exaclty same as kmeans cluster where we try to minimize sum 
#of squares there, here we try to reduce variance with in each cluster
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#fitting hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],color='blue',label = 'cluster-1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],color='red',label = 'cluster-2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],color='green',label = 'cluster-3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],color='violet',label = 'cluster-4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],color='orange',label = 'cluster-5')
plt.title('Clusters of customers')
plt.xlabel('Annaul income')
plt.ylabel('Spending score')
plt.legend()
plt.show()