# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:03:11 2016

@author: samsisto
"""

from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import explore as ex
from textblob import TextBlob, Word
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

###############################################################################
#Naive bayes to predict review and business star rating
###############################################################################
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
print 'Review text vs. Review Stars:'
naive_bayes(ex.texas.text, ex.texas.stars_review)

#Naive bayes on review text vs. business stars
print 'Review text vs. Business Stars:'
naive_bayes(ex.texas.text, ex.texas.stars_business)

###############################################################################
#Naive bayes on categories to predict star ratings
###############################################################################

def naive_bayes_categories(y_value):
    X_test_dtm = ex.categories_dtm.head(56)
    X_train_dtm = ex.categories_dtm.tail(226)
    
    y_test = y_value.head(56)
    y_train = y_value.tail(226)
    
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    
    y_pred_class = nb.predict(X_test_dtm)
    
    print 'Accuracy: '
    print metrics.accuracy_score(y_test, y_pred_class)
    
    print 'Null Accuracy: '
    print 1-sum(y_test)/float(len(y_test))
    
    print 'Confusion Matrix: '
    print metrics.confusion_matrix(y_test, y_pred_class)

#Naive Bayes on categories_dtm vs. city
print 'Category DTM vs. City:'
naive_bayes_categories(ex.tb.city_binary)

#Naive Bayes on categories_dtm vs. stars
print 'Category DTM vs. Review Stars:'
naive_bayes_categories(ex.tb.stars)

###############################################################################
#NLP Analysis
###############################################################################

texas_best_worst = ex.texas[(ex.texas.stars_review==5) | (ex.texas.stars_review==1)]

X = texas_best_worst.text
y = texas_best_worst.stars_review

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
vect = CountVectorizer()
    
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
    
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

print 'Accuracy:'
print metrics.accuracy_score(y_test, y_pred_class)
    
y_test_binary = np.where(y_test==5, 1, 0)
print 'Null Accuracy:'
print max(y_test_binary.mean(), 1 - y_test_binary.mean())

def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print 'Features: ', X_train_dtm.shape[1]
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print 'Accuracy: ', metrics.accuracy_score(y_test, y_pred_class)
    y_test_binary = np.where(y_test==5, 1, 0)
    print 'Null Accuracy: ', max(y_test_binary.mean(), 1 - y_test_binary.mean())

print '1-gram tokens'
vect = CountVectorizer(ngram_range=(1,1))
tokenize_test(vect)

print '1 and 2 gram tokens'
vect = CountVectorizer(ngram_range=(1,2))
tokenize_test(vect)

print '1, 2 and 3 gram tokens'
vect = CountVectorizer(ngram_range=(1,3))
tokenize_test(vect)

print '1 and 2 gram tokens, min_df = 2'
vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
tokenize_test(vect)

def split_into_lemmas(text):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]

print '1 and 2 gram tokens, min_df = 2, split into lemmas analyzer'
vect = CountVectorizer(ngram_range=(1, 2), min_df=2, analyzer=split_into_lemmas)
tokenize_test(vect)

def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

ex.texas['sentiment'] = ex.texas.text.apply(detect_sentiment)

###############################################################################
# LDA
###############################################################################

X = ex.texas.text

stoplist = set(CountVectorizer(stop_words='english').get_stop_words() )
texts = [[word for word in document.lower().split() if word not in stoplist] for document in list(X)]

frequency = defaultdict(int)
for text in texts:
     for token in text:
         frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=2, alpha = 'auto')

lda.show_topics()