# -*- coding: utf-8 -*-
"""
Created on Tue Apr 07 10:30:51 2015

@author: Michele
"""

import numpy as np
from sklearn.utils import check_random_state

def make_multitask_classification(n_samples_per_task = [10]*200,
                                  n_relevant_features = 5,
                                  std_relevant_features = [1.0,0.25,0.1,0.05,0.01],
                                  n_irrelevant_features = 20,
                                  task = 200, random_state=None):
    """Generate a random 2-class multitask classification problem.
       A synthetic data set by generating task parameters w_t from a 
       multi-dimensional Gaussian distribution with zero mean and differnt 
       covariances. These are the relevant dimensions we wish to learn. To 
       these we kept adding up to the irrelevant dimensions which are exactly 
       zero. The examples x_ti are randomly picked from the hypercube [0,1]
       The labels y_ti were computed from the wt and x_ti as 
                           y_ti = <w_t, x_ti> + v
       where v is zero-mean Gaussian with standard deviation equal to 0.1.
    Parameters
    ----------
    n_samples : int, optional (default=[10]*200)
        The list of the number of samples per each task.
    n_relevant_features : int, optional (default=5)
        The total number of relevant features. They are a Gaussian distribution
        with dimension the number of relevant features and std the value
        contained in the list std_relevant_features
    std_relevant_features : list of float (default=[1.0,0.25,0.1,0.05,0.01])
        The std of the relevant features. This list as to be a lenght equal to
        the number of relevant features.
    n_irrelevant_features : int, optional (default = 20)
        The total number of irrelevant features. They are fixed to 0.0.
    task : int, optional (default = 200)
        The number of different tasks
    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features+1]
        The generated samples with the first feature that rapresents the task.
    y : array of shape [n_samples]
        The integer labels for class membership of each sample.
    """
    generator = check_random_state(random_state)
    

    if len(std_relevant_features) != n_relevant_features:
        raise ValueError("len(std_relevant_features) != n_relevant_features")
    if len(n_samples_per_task) != task:
        raise ValueError("len(n_samples_per_task) != task")
        
    n_samples = sum(n_samples_per_task)
    n_features = n_relevant_features + n_irrelevant_features

    # Intialize X and y
    X = np.zeros((n_samples, n_features+1))
    y = np.zeros(n_samples, dtype=np.int)
    
    # Setting up the tasks: from 1 to task
    first_line = []
    counter = 1
    for i in n_samples_per_task:
        for j in range(i):
            first_line.append(counter)
        counter += 1
    X[:,0] = np.array(first_line)
        
    # Generation of the w_t task models
    w = []
    #wbias = [0.0]
    wbias = generator.normal(0.0,std_relevant_features)

    for i in range(task):
        wtmp = np.concatenate([generator.normal(wbias,std_relevant_features),[0.0]*n_irrelevant_features])
        w.append(wtmp)
    
     # Generation of the x_ti examples and y_ti labels BALANCED OVER THE CLASS!
    stepclass = 1
    for i in range(n_samples):
        tryfind = True
        while tryfind: 
            X[i, 1:] = generator.rand(n_features)*2.0 - 1.0
            y[i] = 1 if w[int(X[i,0])-1].dot(X[i, 1:]) + generator.normal(0.0,0.1) >= 0 else -1
            if y[i] == stepclass:
                tryfind = False
                stepclass *= -1

    return X, y, w
    

'''
# Test the toy
X,y = make_multitask_classification(n_samples_per_task=10,
                                    n_relevant_features=5,
                                    std_relevant_features = [1.0,0.25,0.1,0.05,0.01],
                                    n_irrelevant_features=30,
                                    task = 200, shuffle=True, random_state=None)
'''