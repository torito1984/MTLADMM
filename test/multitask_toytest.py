# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:36:15 2015

@author: davidmartinezrego
"""

import sys
sys.path.insert(0, '../')

# Test the algorithm with Toy Problem:
import numpy as np
import matplotlib.pyplot as plt
from multitask_toy_problem_2 import  make_multitask_classification
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import scipy.sparse.csr
import scipy.sparse.csc
import multitask_learning.multitask_admm as mt
import time
from math import log10


t0 = time.time()
random_state = np.random.RandomState(0)
# import some data to play with:

# Number of tasks:
n_of_task = 400
T = n_of_task # For later use

# Number of examples for each task: (automatically balanced over +1 and -1 labels)
n_samples_per_task_list = [16]*n_of_task 
for i in xrange(5):
    n_samples_per_task_list[i] = 1000


# Std of the relevant features:
std_relevant_features = [0.25]*8 #[1.0,0.25,0.1,0.05,0.01]
#std_relevant_features = [10.0]*8 #[1.0,0.25,0.1,0.05,0.01]
n_relevant_features = len(std_relevant_features)

# Number of not useful features:
n_irrelevant_features = 100

X,y = make_multitask_classification(n_samples_per_task=n_samples_per_task_list,
                                    n_relevant_features=n_relevant_features,
                                    std_relevant_features = std_relevant_features,
                                    n_irrelevant_features=n_irrelevant_features,
                                    task = n_of_task, random_state=random_state)
#y = np.array([1 if i == 1 else -1 for i in y])

X = scipy.sparse.csr.csr_matrix(X)

# Creating the Kfolds:
K = 2
task_line = X[:,0].toarray()
task_line.shape = (task_line.shape[0],)
task_line = np.multiply(task_line,y.flat)
kf = StratifiedKFold(task_line, n_folds=K, random_state=random_state)
print 'K-fold with K:',K


# Kfolds stratified with Kfold validations:
K_of_validation = 3
roc_auc_list = []
scores_list = []
for train, test in kf:
    print 'K-fold training...'
    fold_task_line = task_line[train]
    Kf_of_validation = StratifiedKFold(fold_task_line, n_folds=K_of_validation, random_state=random_state)

    # Make use of rearragement of hyperparameters (see Massi 2004).
    # This makes the problem comparable to standard SVM for "all separate" and
    # the results more interpretable

    # Original test
    C = [0.1, 1, 10, 100, 1000, 2000, 3000, 4000] # C = T/2*lambda_1
    mu = [10**i for i in range(-1,10)] # mu = T*lambda_2/lambda_1
   
    #C = [100] # C = T/2*lambda_1
    #mu = [10**6] # mu = T*lambda_2/lambda_1
    
    tuned_parameters = {'C': C, 'mu': mu}
    
    scores = ['roc_auc']
    for score in scores:
        print "# Tuning hyper-parameters for %s" % score
        clf = GridSearchCV(estimator = mt.MTLModel(maxADMMiter = 20, maxSGDiter = 20.0, prec = 1e-5, eta = 1.0),
                           param_grid=tuned_parameters, cv=Kf_of_validation, scoring=score, n_jobs = 1, verbose=1)
        ytrain = y[train]
        ytest = y[test]
        ytrain.shape = (ytrain.shape[0])
        ytest.shape = (ytest.shape[0])

        clf.fit(X[train,:], ytrain)
        print "Best parameters set found on development set:"
        print clf.best_params_
        
        real_best = clf.best_estimator_
        scores_list.append(clf.grid_scores_)
        print real_best
        print "Grid scores on development set:"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

        print "Detailed classification report:"
        print "The model is trained on the full development set."
        print "The scores are computed on the full evaluation set."

 
        y_true, y_pred = ytest, clf.predict(X[test,:])
        print classification_report(y_true, [np.sign(i) for i in y_pred])

    # Reshape label set
    ytrain = y[train]
    ytest = y[test]
    ytrain.shape = (ytrain.shape[0])
    ytest.shape = (ytest.shape[0])
    
    classifier = real_best
    y_score = classifier.fit(X[train,:], ytrain).decision_function(X[test,:])
    roc_auc_list.append(roc_auc_score(ytest, y_score))
    print 'Test with test AUC:',roc_auc_list[-1]
    
print 'K-fold with K=',K,'roc_auc_score:',np.mean(roc_auc_list),'+/-',np.std(roc_auc_list)*2
plt.figure(1)
plt.plot(roc_auc_list)
plt.title('AUC - K-fold')


#score_dict = clf.grid_scores_
scores = [np.mean([d[t][1] for d in scores_list]) for t in range(len(clf.grid_scores_))]
#scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(tuned_parameters['C']),len(tuned_parameters['mu']))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
plt.ylabel('C')
plt.xlabel('log(mu)')
plt.colorbar()
plt.yticks(np.arange(len(tuned_parameters['C'])), tuned_parameters['C'], rotation=45)
plt.xticks(np.arange(len(tuned_parameters['mu'])), [log10(i) for i in tuned_parameters['mu']])
plt.show()

print 'Done!'

