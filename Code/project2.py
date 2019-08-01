# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:43:15 2017

@author: msahr
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt



#%%======================================================================================
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']


twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)

train_num = np.zeros(len(set(twenty_train.target)))
for ii in range(len(set(twenty_train.target))):
    idx = twenty_train.target == ii
    train_num[ii] = sum(idx)
    
categories_names = ['graphics', 'misc', 'PC hardware', 'Mac hardware', 'autos', 'motorcycles', 'baseball', 'hockey']
    
n, bins, patches = plt.hist(twenty_train.target, bins = np.arange(0, 1+len(set(twenty_train.target))), align = 'left' )
plt.xticks(range(len(set(twenty_train.target))), categories_names, fontsize = 32)
plt.yticks(fontsize = 32)
plt.show()
print([ n[ii] for ii in range(len(categories_names))])

#%%================================================================================


categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']



twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer(stop_words = 'english')
X_train_counts = count_vect.fit_transform(twenty_train.data)

terms = count_vect.get_feature_names()
print(X_train_counts.shape)

class_count = np.zeros((len(categories), X_train_counts.shape[1]))
TFxICF = np.zeros((len(categories), X_train_counts.shape[1]))
for ii in range(len(categories)):
    idx = (twenty_train.target == ii)
    class_count[ii] = np.sum(X_train_counts[idx], 0)

t_in_c = np.sum(class_count > 0.5 , 0)   
TFxICF = (0.5 + 0.5*class_count/np.max(class_count, 1)[:,np.newaxis])#*np.log(len(categories)/t_in_c)
TFxICF_srt_idx = np.array(TFxICF).argsort()[:, ::-1]
for ii in range(len(categories)):
    print(categories[ii], [terms[jj] for jj in TFxICF_srt_idx[ii,:10]])
    
    
#%%=============================================================================
k = 50

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']


twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)


TFxIDF_vect = TfidfVectorizer(stop_words = 'english',  min_df = 0.0001)
X_train_TFxIDF = TFxIDF_vect.fit_transform(twenty_train.data)
vocab = TFxIDF_vect.vocabulary_
X_train_TFxIDF = X_train_TFxIDF.transpose()
print(X_train_TFxIDF.shape)
u, s, vt = svds(X_train_TFxIDF, k = k)
#u = u.astype(dtype = float16)
ut = u.T
idx = np.floor(np.linspace(0, X_train_TFxIDF.shape[1], 50))
for ii in range(idx.size-1):
    temp = (X_train_TFxIDF[:,idx[ii]:idx[ii+1]].toarray())
    if ii == 0:        
        Dk_train = ut.dot(temp)
    else:
        Dk_train = np.c_[Dk_train, ut.dot(temp)]

TFxIDF_vect = TfidfVectorizer(vocabulary = vocab)      
X_test_TFxIDF = TFxIDF_vect.fit_transform(twenty_test.data)
X_test_TFxIDF = X_test_TFxIDF.transpose()       
idx = np.floor(np.linspace(0, X_test_TFxIDF.shape[1], 50))
for ii in range(idx.size-1):
    temp = (X_test_TFxIDF[:,idx[ii]:idx[ii+1]].toarray())
    if ii == 0:        
        Dk_test = ut.dot(temp)
    else:
        Dk_test = np.c_[Dk_test, ut.dot(temp)]


#%%=============================================================================
#SVM-Hard Margin
#===============================================================================
from sklearn import svm
from sklearn.metrics import confusion_matrix
ROC_points = 200

class_train = np.copy(twenty_train.target)
class_train[class_train < 4.5] = 1
class_train[class_train > 4.5] = -1
           
class_test = np.copy(twenty_test.target)
class_test[class_test < 4.5] = 1
class_test[class_test > 4.5] = -1

clf_HM = svm.SVC(C = 1000, kernel = 'linear', shrinking = False)

clf_HM.fit(Dk_train.T, class_train)
print(clf_HM.score(Dk_test.T, class_test))

w_HM = clf_HM.coef_
b_HM = clf_HM.intercept_
my_class_train = np.sign(w_HM.dot(Dk_train) + b_HM)

predict_test = clf_HM.predict(Dk_test.T)
confusion_matrix_HM = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Hard Margin: ', confusion_matrix_HM)
print('Precision: ', confusion_matrix_HM[0,0]/sum(confusion_matrix_HM[:,0]), 'Accuracy: ',
      (confusion_matrix_HM[0,0] + confusion_matrix_HM[1,1])/np.sum(confusion_matrix_HM.flatten()), 
      'Recall: ', confusion_matrix_HM[0,0]/np.sum(confusion_matrix_HM[0,:]))

b1 = np.linspace(-10, 10, ROC_points)
b1 = b1 + b_HM
test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
false_positive_rate_HM = np.zeros(ROC_points)
true_positive_rate_HM = np.zeros(ROC_points)
for ii in range(len(b1)):
    classes = np.sign(w_HM.dot(Dk_test) + b1[ii])[0]
    positives = (classes > 0)
    true_positives_HM = np.logical_and(positives, test_positives)
    false_positives_HM = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_HM[ii] = np.sum(true_positives_HM)/test_positives_num
    false_positive_rate_HM[ii] = np.sum(false_positives_HM)/test_positives_num
                       
plt.figure()
plt.plot(false_positive_rate_HM, true_positive_rate_HM, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('False Positve Rate', fontsize = 35)
plt.ylabel('True Positive Rate', fontsize = 35)
plt.title('Hard Margin SVM', fontsize = 35)
                      
#%%============================================================================
#SVM- Soft Margin
#==============================================================================

from sklearn import svm
ROC_points = 200     
    

class_train = np.copy(twenty_train.target)
class_train[class_train < 4.5] = 1
class_train[class_train > 4.5] = -1
           
class_test = np.copy(twenty_test.target)
class_test[class_test < 4.5] = 1
class_test[class_test > 4.5] = -1



# 5 fold cross validation for finding the best C in SVM
from sklearn.model_selection import KFold
C = [10**k for k in range(-3, 4)]          
all_data = np.r_[Dk_train.T, Dk_test.T]
all_y = np.concatenate((class_train, class_test))

kf = KFold(n_splits = 5, shuffle = True)

accuracy = np.zeros((5, len(C)))
jj = 0
for gamma in C:
    ii = 0
    for train_index, test_index in kf.split(all_data):
        clf_SM = svm.SVC(C = gamma, kernel = 'linear', shrinking = False)
        clf_SM.fit(all_data[train_index], all_y[train_index])
        accuracy[ii, jj] = clf_SM.score(all_data[test_index], all_y[test_index])
        ii +=1
    jj += 1
        
print(accuracy)
plt.figure()
plt.plot(C, np.mean(accuracy, 0), linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('Lambda', fontsize = 35)
plt.ylabel('Accuracy', fontsize = 35)
plt.title('Regularization effect on accuracy', fontsize = 35)


idx = np.argmax(np.mean(accuracy, 0))
gamma = C[idx]
clf_SM = svm.SVC(C = gamma, kernel = 'linear', shrinking = False)

clf_SM.fit(Dk_train.T, class_train)
print(clf_SM.score(Dk_test.T, class_test))
predict_test = clf_SM.predict(Dk_test.T)
confusion_matrix_SM = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Soft Margin SVM: ', confusion_matrix_SM)
print('Precision: ', confusion_matrix_SM[0,0]/sum(confusion_matrix_SM[:,0]), 'Accuracy: ',
      (confusion_matrix_SM[0,0] + confusion_matrix_SM[1,1])/np.sum(confusion_matrix_SM.flatten()), 
      'Recall: ', confusion_matrix_SM[0,0]/np.sum(confusion_matrix_SM[0,:]))

w_SM = clf_SM.coef_
b_SM = clf_SM.intercept_

b1 = np.linspace(-10, 10, ROC_points)
b1 = b1 + b_SM
test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
false_positive_rate_SM = np.zeros(ROC_points)
true_positive_rate_SM = np.zeros(ROC_points)
for ii in range(len(b1)):
    classes = np.sign(w_SM.dot(Dk_test) + b1[ii])[0]
    positives_SM = (classes > 0)
    true_positives_SM = np.logical_and(positives_SM, test_positives)
    false_positives_SM = np.logical_and(positives_SM, np.logical_not(test_positives))
    true_positive_rate_SM[ii] = np.sum(true_positives_SM)/test_positives_num
    false_positive_rate_SM[ii] = np.sum(false_positives_SM)/test_positives_num
                       
plt.figure()
plt.plot(false_positive_rate_SM, true_positive_rate_SM, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('False Positve Rate', fontsize = 35)
plt.ylabel('True Positive Rate', fontsize = 35)
plt.title('Soft Margin SVM', fontsize = 35)
#%%============================================================================
#Naive Bayes
#==============================================================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
ROC_points = 200

class_train = np.copy(twenty_train.target)
class_train[class_train < 4.5] = 1
class_train[class_train > 4.5] = -1
           
class_test = np.copy(twenty_test.target)
class_test[class_test < 4.5] = 1
class_test[class_test > 4.5] = -1

clf_NB = GaussianNB()
clf_NB.fit(Dk_train.T, class_train)
#clf_NB.fit(X_train_TFxIDF.transpose(), class_train)

#predict_test = clf_NB.predict(X_test_TFxIDF.transpose())
predict_test = clf_NB.predict(Dk_test.T, class_test)
confusion_matrix_NB = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Naive Bayes: ', confusion_matrix_NB)
print('Precision: ', confusion_matrix_NB[0,0]/sum(confusion_matrix_NB[:,0]), 'Accuracy: ',
      (confusion_matrix_NB[0,0] + confusion_matrix_NB[1,1])/np.sum(confusion_matrix_NB.flatten()), 
      'Recall: ', confusion_matrix_NB[0,0]/np.sum(confusion_matrix_NB[0,:]))

p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
false_positive_rate_NB = np.zeros(ROC_points)
true_positive_rate_NB = np.zeros(ROC_points)
#class_probs = clf_NB.predict_proba(X_test_TFxIDF.transpose())
class_probs = clf_NB.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    
    positives = (class_probs[:,0] < p1[ii])
    true_positives_NB = np.logical_and(positives, test_positives)
    false_positives_NB = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_NB[ii] = np.sum(true_positives_NB)/test_positives_num
    false_positive_rate_NB[ii] = np.sum(false_positives_NB)/test_positives_num
                       
plt.figure()
plt.plot(false_positive_rate_NB, true_positive_rate_NB, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('False Positve Rate', fontsize = 35)
plt.ylabel('True Positive Rate', fontsize = 35)
plt.title('Naïve Bayes', fontsize = 35)


#%%============================================================================
#Logistic Regression Classifier
#==============================================================================
from sklearn.linear_model import LogisticRegression
ROC_points = 150

class_train = np.copy(twenty_train.target)
class_train[class_train < 4.5] = 1
class_train[class_train > 4.5] = -1
           
class_test = np.copy(twenty_test.target)
class_test[class_test < 4.5] = 1
class_test[class_test > 4.5] = -1

clf_LG = LogisticRegression(C = 1000)

clf_LG.fit(Dk_train.T, class_train)

predict_test = clf_LG.predict(Dk_test.T)
confusion_matrix_LG = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Logistic Regression: ', confusion_matrix_LG)
print('Precision: ', confusion_matrix_LG[0,0]/sum(confusion_matrix_LG[:,0]), 'Accuracy: ',
      (confusion_matrix_LG[0,0] + confusion_matrix_LG[1,1])/np.sum(confusion_matrix_LG.flatten()), 
      'Recall: ', confusion_matrix_LG[0,0]/np.sum(confusion_matrix_LG[0,:]))

p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
false_positive_rate_LG = np.zeros(ROC_points)
true_positive_rate_LG = np.zeros(ROC_points)
class_probs = clf_LG.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    
    positives = (class_probs[:,0] < p1[ii])
    true_positives_LG = np.logical_and(positives, test_positives)
    false_positives_LG = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_LG[ii] = np.sum(true_positives_LG)/test_positives_num
    false_positive_rate_LG[ii] = np.sum(false_positives_LG)/test_positives_num
                       
plt.figure()
plt.plot(false_positive_rate_LG, true_positive_rate_LG, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('False Positve Rate', fontsize = 35)
plt.ylabel('True Positive Rate', fontsize = 35)
plt.title('Logistic Regression', fontsize = 35)


#%%============================================================================
#Logistic Regression with L1 regularizer
#==============================================================================
from sklearn.linear_model import LogisticRegression
ROC_points = 100

class_train = np.copy(twenty_train.target)
class_train[class_train < 4.5] = 1
class_train[class_train > 4.5] = -1
           
class_test = np.copy(twenty_test.target)
class_test[class_test < 4.5] = 1
class_test[class_test > 4.5] = -1

C = 10**(np.linspace(-3, 2, 50))


accuracy_LG_L1 = np.zeros(C.size)

for ii in range(len(C)):
    clf_LG_L1 = LogisticRegression(penalty = 'l1', C = 1/C[ii])
    clf_LG_L1.fit(Dk_train.T, class_train)
    accuracy_LG_L1[ii] = clf_LG_L1.score(Dk_test.T, class_test)
            
plt.figure()
plt.plot(C, accuracy_LG_L1, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('Regularization Coefficient', fontsize = 35)
plt.ylabel('Accuracy', fontsize = 35)
plt.title('Logistic Regression with L1 regularizer', fontsize = 35)


idx = np.argmax(accuracy_LG_L1)
C = C[idx]

clf_LG_L1 = LogisticRegression(penalty = 'l1', C = 1/C)

clf_LG_L1.fit(Dk_train.T, class_train)
predict_test = clf_LG_L1.predict(Dk_test.T)
confusion_matrix_LG_L1 = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Soft Margin Logistic Regression with L1 Penalty: ', confusion_matrix_LG_L1)
print('Precision: ', confusion_matrix_LG_L1[0,0]/sum(confusion_matrix_LG_L1[:,0]), 'Accuracy: ',
      (confusion_matrix_LG_L1[0,0] + confusion_matrix_LG_L1[1,1])/np.sum(confusion_matrix_LG_L1.flatten()), 
      'Recall: ', confusion_matrix_LG_L1[0,0]/np.sum(confusion_matrix_LG_L1[0,:]))

p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
false_positive_rate_LG_L1 = np.zeros(ROC_points)
true_positive_rate_LG_L1 = np.zeros(ROC_points)
class_probs = clf_LG_L1.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    
    positives = (class_probs[:,0] < p1[ii])
    true_positives_LG_L1 = np.logical_and(positives, test_positives)
    false_positives_LG_L1 = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_LG_L1[ii] = np.sum(true_positives_LG_L1)/test_positives_num
    false_positive_rate_LG_L1[ii] = np.sum(false_positives_LG_L1)/test_positives_num
                       
plt.figure()
plt.plot(false_positive_rate_LG_L1, true_positive_rate_LG_L1, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('False Positve Rate', fontsize = 35)
plt.ylabel('True Positive Rate', fontsize = 35)
plt.title('Logistic Regression with L1 regularization', fontsize = 35)

#%%============================================================================
#Logistic Regression with L2 regularizer
#==============================================================================
from sklearn.linear_model import LogisticRegression
ROC_points = 100
class_train = np.copy(twenty_train.target)
class_train[class_train < 4.5] = 1
class_train[class_train > 4.5] = -1
           
class_test = np.copy(twenty_test.target)
class_test[class_test < 4.5] = 1
class_test[class_test > 4.5] = -1

C = 10**(np.linspace(-3, 2, 50))


accuracy_LG_L2 = np.zeros(C.size)

for ii in range(len(C)):
    clf_LG_L2 = LogisticRegression(penalty = 'l2', C = 1/C[ii])
    clf_LG_L2.fit(Dk_train.T, class_train)
    accuracy_LG_L2[ii] = clf_LG_L2.score(Dk_test.T, class_test)
            
plt.figure()
plt.plot(C, accuracy_LG_L2, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('Regularization Coefficient', fontsize = 35)
plt.ylabel('Accuracy', fontsize = 35)
plt.title('Logistic Regression with L2 regularizer', fontsize = 35)

idx = np.argmax(accuracy_LG_L2)
C = C[idx]

clf_LG_L2 = LogisticRegression(C = 1/C)

clf_LG_L2.fit(Dk_train.T, class_train)
predict_test = clf_SM.predict(Dk_test.T)
confusion_matrix_LG_L2 = confusion_matrix(class_test, predict_test)
print('Confuxion Matrix for Logistic Regression with L2 Penalty: ', confusion_matrix_LG_L2)

p1 = np.linspace(0, 1, ROC_points)

test_positives = (class_test > 0)
test_positives_num = sum(test_positives)
false_positive_rate_LG_L2 = np.zeros(ROC_points)
true_positive_rate_LG_L2 = np.zeros(ROC_points)
class_probs = clf_LG_L2.predict_proba(Dk_test.T)
for ii in range(len(p1)):
    
    positives = (class_probs[:,0] < p1[ii])
    true_positives_LG_L2 = np.logical_and(positives, test_positives)
    false_positives_LG_L2 = np.logical_and(positives, np.logical_not(test_positives))
    true_positive_rate_LG_L2[ii] = np.sum(true_positives_LG_L2)/test_positives_num
    false_positive_rate_LG_L2[ii] = np.sum(false_positives_LG_L2)/test_positives_num
                       
plt.figure()
plt.plot(false_positive_rate_LG_L2, true_positive_rate_LG_L2, linewidth = 2)
plt.xticks(fontsize = 32)
plt.yticks(fontsize = 32)
plt.xlabel('False Positve Rate', fontsize = 35)
plt.ylabel('True Positive Rate', fontsize = 35)
plt.title('Logistic Regression with L2 regularization', fontsize = 35)


#%%============================================================================
#Multiclass Naive Bayes
#==============================================================================
from sklearn.metrics import recall_score, precision_score

categories = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

k = 50

twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)

TFxIDF_vect = TfidfVectorizer(stop_words = 'english',  min_df = 0.0001)
X_train_TFxIDF = TFxIDF_vect.fit_transform(twenty_train.data)
vocab = TFxIDF_vect.vocabulary_
X_train_TFxIDF = X_train_TFxIDF.transpose()

u, s, vt = svds(X_train_TFxIDF, k = k)

ut = u.T
idx = np.floor(np.linspace(0, X_train_TFxIDF.shape[1], 50))
for ii in range(idx.size-1):
    temp = (X_train_TFxIDF[:,idx[ii]:idx[ii+1]].toarray())
    if ii == 0:        
        Dk_train = ut.dot(temp)
    else:
        Dk_train = np.c_[Dk_train, ut.dot(temp)]

TFxIDF_vect = TfidfVectorizer(vocabulary = vocab)      
X_test_TFxIDF = TFxIDF_vect.fit_transform(twenty_test.data)
X_test_TFxIDF = X_test_TFxIDF.transpose()       
idx = np.floor(np.linspace(0, X_test_TFxIDF.shape[1], 50))
for ii in range(idx.size-1):
    temp = (X_test_TFxIDF[:,idx[ii]:idx[ii+1]].toarray())
    if ii == 0:        
        Dk_test = ut.dot(temp)
    else:
        Dk_test = np.c_[Dk_test, ut.dot(temp)]
        
        
from sklearn.naive_bayes import MultinomialNB
ROC_points = 200


clf_NB = MultinomialNB()

clf_NB.fit(X_train_TFxIDF.transpose(), twenty_train.target)

predict_test = clf_NB.predict(X_test_TFxIDF.transpose())
confusion_matrix_NB = confusion_matrix(twenty_test.target, predict_test)
print('Confuxion Matrix for Naive Bayes: ', confusion_matrix_NB)
print('Precision: ', precision_score(twenty_test.target, predict_test, average = 'macro'), 'Accuracy: ',
      precision_score(twenty_test.target, predict_test, average = 'micro'), 'Recall: ', recall_score(twenty_test.target, predict_test, average = 'macro'))

#%%============================================================================
#SVM- One vs One
#==============================================================================

clf_OVO = svm.SVC(C = 1000, kernel = 'linear', shrinking = False)

clf_OVO.fit(Dk_train.T, twenty_train.target)

predict_test = clf_OVO.predict(Dk_test.T)
confusion_matrix_OVO = confusion_matrix(twenty_test.target, predict_test)
print('Confuxion Matrix for SVM OVO: ', confusion_matrix_NB)
print('Precision: ', precision_score(twenty_test.target, predict_test, average = 'macro'), 'Accuracy: ',
      precision_score(twenty_test.target, predict_test, average = 'micro'), 'Recall: ', recall_score(twenty_test.target, predict_test, average = 'macro'))

#SVM- one vs rest
clf_OVR = svm.LinearSVC(C = 100, multi_class = 'ovr')

clf_OVR.fit(Dk_train.T, twenty_train.target)

predict_test = clf_OVR.predict(Dk_test.T)
confusion_matrix_OVR = confusion_matrix(twenty_test.target, predict_test)
print('Confuxion Matrix for SVM OVR: ', confusion_matrix_NB)
print('Precision: ', precision_score(twenty_test.target, predict_test, average = 'macro'), 'Accuracy: ',
      precision_score(twenty_test.target, predict_test, average = 'micro'), 'Recall: ', recall_score(twenty_test.target, predict_test, average = 'macro'))
