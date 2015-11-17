# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:45:08 2015

@author: Michele Donini
"""
 
from Cmultitask_admm import CtrainTask
from Cmultitask_admm import CfindZ
from Cmultitask_admm import CdecisionFunction
from Cmultitask_admm import init

import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.sparse.coo
from scipy import sparse
import scipy.sparse.csr
#from scipy.sparse import identity

class MTLModel(BaseEstimator, ClassifierMixin):
    """Multitask ADMM"""
    def __init__(self, maxADMMiter = 10, maxSGDiter = 2, prec = 1e-5, C = 1.0, mu = 1.0, eta = 1.0):
        self.maxADMMiter = maxADMMiter
        self.maxSGDiter = maxSGDiter
        self.prec = prec

        self.C = C
        self.mu = mu
        self.ro1 = None
        self.ro2 = None
        self.eta = eta
        
        self.W = None
        self.D2 = None
        self.R = None
        self.beta = None
        self.M = None
        self.T = None

        self.outer_step = 0

        self.verbose = False
    
    
    def fit(self, X, labels):
        ''' 
            The first feature of an example x_i indicates the TASK of the example
            X = [ex1_task1, ..., exm1_tast1, ex1_task2, ..., exm2_tast2, ..., ex1_taskT, ..., exmT_taskT]
            labels = [y1_task1, ..., ym1_tast1, y1_task2, ..., ym2_tast2, ..., y1_taskT, ..., ymT_taskT]
        '''
        # Group data by task
        if type(X) == scipy.sparse.coo.coo_matrix or type(X) == scipy.sparse.csr.csr_matrix:
            Xfirst_line = X[:,0].toarray()
            maxfirst = int(np.amax(Xfirst_line))
        else:
            Xfirst_line = [X[i][0] for i in range(X.shape[0])]
            maxfirst = int(np.max(Xfirst_line))

        self.M_list = [0]*maxfirst
        self.Xtasks = [list() for _ in xrange(maxfirst)]

        for idx in xrange(len(Xfirst_line)):
            t = int(Xfirst_line[idx]) - 1 # Array zero based, tasks one based
            self.M_list[t] += 1
            self.Xtasks[t].append(idx)
        
        # Clear the task id, not needed anymore --> Ignore task inside the training function
        #a = sparse.eye(X.shape[1])
        #a.setdiag([0]) # Delete first column
        #X = X*a
        
        # Doubt: this is actually R^2
        # Radius of the ball of the examples
        self.R = X[:,1:].multiply(X[:,1:]).sum(axis = 1).max()


        # Radius of the ball of the w and z
        self.M = float(sum(self.M_list))
        self.T = float(len(self.M_list))
        
        # Set parameters
        self.lam1 = float(self.T)/(2*self.C)
        self.lam2 = (self.mu*self.lam1)/float(self.T)
        
        self.ro1 = (1.0/self.T) * (self.lam1 * self.lam2) / (self.lam1 + self.lam2)
        self.ro2 = (1.0/self.T) * (self.lam2 * self.lam2) / (self.lam1 + self.lam2)

        if self.verbose:
            print 'Radius of the examples R:',self.R
            
        lam1 = self.lam1 #(self.ro1 * (self.ro1 + self.ro2) * self.T) / self.ro2 
        self.beta = self.T * self.M / ( 2 * lam1)
        
        if self.verbose:
            print 'lam_1:',self.lam1,
            print 'lam_2:',self.lam2
            print 'ro_1:',self.ro1,
            print 'ro_2:',self.ro2
        
        self.W = np.zeros(len(self.M_list) * X.shape[1], dtype=np.float32)
        Y = np.zeros(len(self.M_list) * X.shape[1], dtype=np.float32)
        Z = np.zeros(len(self.M_list) * X.shape[1], dtype=np.float32)
        Yold = np.zeros(len(self.M_list) * X.shape[1], dtype=np.float32)
        Zold = np.zeros(len(self.M_list) * X.shape[1], dtype=np.float32)
        Yhat = np.zeros(len(self.M_list) * X.shape[1], dtype=np.float32)
        Zhat = np.zeros(len(self.M_list) * X.shape[1], dtype=np.float32)
        alphaold = 1.0
        alphanew = 1.0
        
        # Initialize space for optimization
        init(int(X.shape[1]))
        
        # Fit the model => ADMM outer iteration
        for it in xrange(self.maxADMMiter):
            if self.verbose:
                print 'ADMM step:',it+1
            self.outer_step = it + 1
            
            # W optimization
            for task in xrange(len(self.M_list)): # Zero based task indexing
                if self.verbose:
                    print 'Start optimizing task:',task,'|',len(self.M_list)  
                self.W[X.shape[1] * task : X.shape[1] * (task+1)] = self.trainTask(X,task,labels,Z, Y)


            if self.verbose:
                print 'Start optimizing Z'
            self.findZ(self.W,Y,Z)
            
            if self.verbose:
                print 'Start finding Y'
            Y = self.findY(Y,self.W,Z)

            # By removing the correction, we recover Bersektas algorithm
            # calculate c_k
            # alphanew = (1.0+np.sqrt(1.0 + 4.0 * alphanew**2))/2.0
            # Zhat = Z #+ (alphaold - 1.0)/alphanew * (Z-Zold)
            # Yhat = Y #+ (alphaold - 1.0)/alphanew * (Y-Yold)
            # alphaold = alphanew
            # Zold = Z
            # Yold = Y
        return self
        
    def predict(self, X):
        return self.decision_function(X)

    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"maxADMMiter": self.maxADMMiter, "maxSGDiter": self.maxSGDiter, "prec": self.prec,
                "C": self.C, "mu": self.mu, "eta": self.eta}

    def set_params(self, **parameters):
        #print parameters
        for parameter, value in parameters.items():
            #print parameter,value
            setattr(self,parameter,value)
            #self.__setattr__(parameter, value)
            #self.setattr(parameter, value)
        return self
    
    def decision_function(self, X):
        ''' Distance of the samples X to the separating hyperplane.
            Parameters:	
                X : array-like, shape = [n_samples, n_features]
            Returns:	
                X : array-like, shape = [n_samples, n_class * (n_class-1) / 2]
                Returns the decision function of the sample for each class in the model.
        '''
        return CdecisionFunction(X.indices,X.indptr,X.data,X.shape[0],X.shape[1],
                                     self.W, self.T, X.shape[1])
    
    def trainTask(self,X,t,labels,totZ,totY):
        '''for el in [self.M,self.T,self.lam1,self.prec,
                          self.maxSGDiter,self.ro1,self.eta,
                          self.Xtasks[t],self.beta,float(self.R),self.outer_step,
                          self.verbose,X.indices,X.indptr,X.data,X.shape[0],X.shape[1],t,labels,totZ,totY]:
            print type(el)
        '''
        return CtrainTask(self.M,self.T,
                          self.lam1,
                          self.prec,
                          self.maxSGDiter,
                          self.ro1,
                          self.eta,
                          self.Xtasks[t],
                          self.beta,
                          float(self.R),
                          self.outer_step,
                          self.verbose,
                          X.indices,
                          X.indptr,
                          X.data,
                          X.shape[0],
                          X.shape[1],
                          t,
                          labels,
                          totZ,
                          totY)
            

    def findZ(self,w,y,z):
        return CfindZ(0,
                      self.T,
                      self.lam1,
                      self.ro1,
                      self.ro2,
                      self.prec,
                      self.maxSGDiter,
                      self.eta,
                      self.beta,
                      self.outer_step,
                      self.verbose,
                      w,
                      y,
                      z)


    def findY(self,y,w,z):
        ''' .... '''
        return y + self.eta * (w - z)
    
    def findc(self,x,z,zhat):
        return  self.eta * np.linalg.norm(z-x)**2 + self.eta * np.linalg.norm(z-zhat)**2