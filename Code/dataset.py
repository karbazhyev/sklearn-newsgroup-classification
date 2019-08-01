# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:03:05 2017

@author: msahr
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

train_num = np.zeros(len(set(twenty_train.target)))
for ii in range(len(set(twenty_train.target))):
    idx = twenty_train.target == ii
    train_num[ii] = sum(idx)
    

    
n, bins, patches = plt.hist(twenty_train.target, bins = np.arange(0, 1+len(set(twenty_train.target))), align = 'left' )
plt.xticks(range(len(set(twenty_train.target))), twenty_train.target_names, fontsize = 16)
plt.yticks(fontsize = 32)
plt.show()



vectorizer = CountVectorizer(min_df=1)

stop_words = text.ENGLISH_STOP_WORDS
print(stop_words)
print(len(stop_words))

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)


vectorizer.get_feature_names() == (
    ['and', 'document', 'first', 'is', 'one',
     'second', 'the', 'third', 'this'])


X.toarray()

vectorizer.vocabulary_.get('document')

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

list(count_vect.vocabulary_.keys())[0:10]

from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


print (list(count_vect.vocabulary_.keys())[0:10])
print('\n')
print (take(5,count_vect.vocabulary_.items()))
print('\n')
print (count_vect.vocabulary_.get('polytopes'))
print(count_vect.vocabulary_['150mb'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
X_train_tfidf.toarray()[:30,:10]

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target) # train

docs_new = ['He is an OS developer', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
    
    
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
    
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
print(docs_test[0])
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target) 

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))

metrics.confusion_matrix(twenty_test.target, predicted)

print(train_num)