# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:36:15 2015

@author: davidmartinezrego
"""


# Test the algorithm with Toy Problem:
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  datasets
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
import scipy.sparse.csr
import scipy.sparse.csc
import multitask_learning.multitask_admm as mt
import time
import sys

t0 = time.time()
random_state = np.random.RandomState(0)
# import some data to play with:
X,y = datasets.make_classification(n_samples=20, n_features=10, n_informative=10, n_redundant=0, n_repeated=0, n_classes=2,
                                   n_clusters_per_class=1, weights=None, flip_y=0.0, class_sep=10.0, hypercube=True,
                                   shift=0.0, scale=1.0, shuffle=True, random_state=random_state)
y = np.array([-1 if i == 0 else 1 for i in y])

# fix manually the T tasks:
T =500
print 'Tasks:',T
task_feat = np.array([i%T + 1 for i in xrange(X.shape[0])]) # Task count between 1 and max
task_feat.shape = (X.shape[0],1)
X = np.concatenate((task_feat,X),1)
X = scipy.sparse.csr.csr_matrix(X)

# Split into training and test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=random_state)
print 'Train Set:',y_train.shape[0]
print 'Test Set:',y_test.shape[0]

#print 'Start at:',time.time()-t0
classifier = mt.MTLModel(maxADMMiter = 10, maxSGDiter = 1.2, prec = 1e-5, lam1 = 1.0, lam2 = 1.0, eta = 1.0)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
print 'Trained at:',time.time()-t0
print 'roc_auc_score with 50% train randomly selected:', roc_auc_score(y_test, [np.sign(i) for i in y_score])

'''
# K-fold:
K = 2
kf = KFold(X.shape[0], n_folds=K, random_state=random_state)
roc_auc_list = []
for train, test in kf:
    classifier = mt.MTLModel(maxADMMiter = 10, maxSGDiter = 1.2, prec = 1e-5, lam1 = 1.0, lam2 = 1.0, eta = 1.0)
    y_score = classifier.fit(X[train][:], y[train]).decision_function(X[test][:])
    #print 'Fold with train examples for task:',[[X[i][0] for i in train].count(j) for j in range(T)],
    roc_auc_list.append(roc_auc_score(y[test], y_score))
    print 'with test AUC:',roc_auc_list[-1]
print 'K-fold with K=',K,'roc_auc_score:',np.mean(roc_auc_list),'+/-',np.std(roc_auc_list)*2
plt.plot(roc_auc_list)
plt.title('AUC - K-fold')
#plt.show(block=False)
#plt.show()
'''


tuned_parameters = [{'lam1': [0.1, 0.3, 1.0, 3.0, 10.0, 1e8], 'lam2': [0.1, 0.3, 1.0, 3.0, 10.0, 1e8]}]
scores = ['roc_auc']
for score in scores:
    print "# Tuning hyper-parameters for %s" % score
    clf = GridSearchCV(mt.MTLModel(maxADMMiter = 10, maxSGDiter = 1.2, prec = 1e-5, lam1 = 1.0, lam2 = 1.0, eta = 1.0),
                       tuned_parameters, cv=2, scoring=score, verbose=1)
    clf.fit(X_train, y_train)
    print "Best parameters set found on development set:"
    print clf.best_estimator_
    print "Grid scores on development set:"
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

    print "Detailed classification report:"
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, [np.sign(i) for i in y_pred])

print 'Done!'

