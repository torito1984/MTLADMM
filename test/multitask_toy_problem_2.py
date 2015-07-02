# -*- coding: utf-8 -*-
"""
Created on Tue Apr 07 10:30:51 2015

@author: Michele
"""

import numpy as np
import math as math
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
    # One extra in X for the task
    X = np.zeros((n_samples, n_features+1))
    y = np.zeros(n_samples, dtype=np.int)
    

    margin  = 5    # Margin of the classification tasks
    vmargin = 0.1  # Variance in the normal to the classification hyperplane
    ovar    = 20   # Variance in the orthogonal direction of the normal vector for the datapoints
    ivar    = 10   # Variance in the orthogonal direction of the normal vector for the datapoints
    pchange = 2    # Propotion of the change in values of the regressor (compared to the original value)
    pnchange = 0.6 # Propotion of features that change in the regressor (compared to the original value)

    # Generation of the w_t task models
    wbase = generator.normal(0.0, std_relevant_features)
    wbase = wbase/np.linalg.norm(wbase) # Normalize
    
    # Calculate number of variables that change
    nchange   = int(math.floor(pnchange*n_relevant_features))

    # Counters for generating patterns
    tcounter = 1
    pat      = 0

    # Variances of data in the irrelevant features
    std_irrelevant = np.absolute(generator.normal(0.0, [ivar]*n_irrelevant_features))

    variables2change = []
    rr = range(1, n_relevant_features)
    for el in xrange(nchange):
        variables2change.append(generator.choice(rr))
    
    for i in n_samples_per_task:
        wtmp = np.concatenate([wbase, [0.0]*n_irrelevant_features])
        # Modify vector (Uncomment this for multitask1
        for changevar in variables2change:
            wtmp[changevar] = wtmp[changevar] + generator.normal(0.0, pchange*math.fabs(wtmp[changevar]))
            wtmp = wtmp/np.linalg.norm(wtmp) # Normalize

        # Generate data
        stepclass = 1
        for j in range(i):
            X[pat,0] = tcounter
            # Calculate a orthogonal direction
            oo = generator.normal(0.0, [ovar]*(n_relevant_features-1))
            k  = np.dot(oo, wtmp[1:n_relevant_features])
            orthogonal = np.concatenate([[-k/wtmp[0]], oo])
            orthogonal = orthogonal/np.linalg.norm(orthogonal) # Normalize
            
            # Generate pattern
            y[pat] = stepclass
            X[pat, 1:] =  np.concatenate([(margin*y[pat] + generator.normal(0.0, vmargin))*wtmp[0:n_relevant_features] + generator.normal(0.0, ovar)*orthogonal,  generator.normal(0.0, std_irrelevant)])
            pat += 1
            stepclass *= -1
        tcounter += 1

    # Displace everything from the origin - We cannot do this wihouth bias in the model
    #std_movement = np.concatenate([np.multiply(std_relevant_features,5), std_irrelevant])
    #wbias = generator.normal(10.0,std_movement)
    #for i in range(n_samples):
    #    X[i, 1:] = X[i, 1:] + wbias

    return X, y
