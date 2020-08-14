import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter ='\t',quoting = 3)

#cleaning the texts
import re

#using sub function we try to eliminate all the numbers from the review
#1st param : it tells we are not going to remove characters that are a-z and A-Z
#2nd param: if we remove a character from the review and the characters to the left and right of it are going to stuck together and doesn't make any sense
#           it create a word that doesn't make any sense, so now removed character is replaced by ' '(space)

review = re.sub('[^a-zA-z]',' ',dataset['Review'][0])
#converting all characters to lower cases
review= review.lower()

#removing words like prepositions,articles which doesn't make any impact on the review which tells bad or good
import nltk
nltk.download('stopwords')
#this stopwords contains all the words which doesn't tell whether the review is bad or good, simply we don't collect those words in our word bag
#now we loop over our review, and if the word is present in stopwords then we eliminate that word
#we convert string to list so iteration becomes easier
review = review.split()
#importing stopwords package
from nltk.corpus import stopwords
review = [word for word in review if not word in set(stopwords.words('english'))]

#Stemming, we simply apply stemming function to each word, this step can be performed parallely to the above step

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#now we join all words of the list to a string
#here ' ' implies there will be space appended between the words
review = ' '.join(review)

#we have done for only one review, now we are going to do this for all 1000 review
corpus=[]
corpus.append(review) #since we have already done it for the 1st review
for i in range(1,1000):
    review = re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review= review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #use of this is we can have a very huge number of colums in our X matrix which leads to sparcity
X = cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values

#From now we use naive bayes or decision tree classification
#we are using naive bayes here

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
Accuracy= accuracy_score(y_test,y_pred)
    