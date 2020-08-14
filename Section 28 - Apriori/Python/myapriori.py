import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#our dataset has no header
dataset= pd.read_csv('Market_Basket_Optimisation.csv',header=None)

#apriori algorithm take input as lists of lists, so we
#are converting our dataframe into lists of lists
transactions=[]
for i in range(0,7501):
    st=[]
    for j in range(0,20):
        st.append(str(dataset.values[i,j]))
    transactions.append(st)
"""for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])"""

#Training apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#visualizing the results
results=list(rules)

result= str(results)
frozensets=result.split("RelationRecord")