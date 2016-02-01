# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:03:11 2016

@author: samsisto
"""

from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import explore as ex

def naive_bayes(x_value, y_value):
    X = x_value
    y = y_value

    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)

    vect = CountVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)

    X_test_dtm = vect.transform(X_test)

    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    
    print 'Accuracy: '
    print metrics.accuracy_score(y_test, y_pred_class)
    
    print 'Null Accuracy: '
    print y_test.value_counts().head(1) / len(y_test)
    
    print 'Confusion Matrix: '
    print metrics.confusion_matrix(y_test, y_pred_class)

ex.texas['stars_business'] = np.round(ex.texas.stars_business, decimals=0)

#Naive bayes on review text vs. review stars
print 'Review test vs. Review Stars: '
naive_bayes(ex.texas.text, ex.texas.stars_review)

#Naive bayes on review text vs. business stars
print 'Review test vs. Business Stars: '
naive_bayes(ex.texas.text, ex.texas.stars_business)