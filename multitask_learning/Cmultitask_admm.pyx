# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:23:38 2015

@author: Michele
"""

import cython
import numpy as np
cimport numpy as np

# Native C libraries

from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

#cdef extern from "src/cblas/cblas.h":
#    double ddot "cblas_sdot"(int, float *, int, float *, int) nogil
#    void dscal "cblas_sscal"(int, float, float *, int) nogil,
#    double dnrm2 "cblas_snrm2" (int, float *, int) nogil

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
FFLOAT = np.float32
DDOUBLE = np.double
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.float32_t FLOAT_t
ctypedef np.double_t DOUBLE_t


# Reserve temporal variables for optimization
cdef double * tmpdata

def init(int dim):
    global tmpdata
    tmpdata = <double *>malloc(dim*cython.sizeof(double))

@cython.boundscheck(False)
def CtrainTask(float M,
              float T,
              float selfobjlam1,
              float selfobjprec,
              int selfobjmaxSGDiter,
              float selfobjro1,
              float selfobjeta,
              selfobjXtasksPy,
              float selfobjbeta,
              float selfobjR,
              int selfobjouter_step,
              bint selfobjverbose, #boolean as int
              np.ndarray[cython.int, ndim=1] Xindices,
              np.ndarray[cython.int, ndim=1] Xindptr,
              np.ndarray[cython.double, ndim=1] Xdata,
              int Xshape0,
              int Xshape1,
              int t,
              np.ndarray[long] labels,
              np.ndarray[float] totZ,
              np.ndarray[float] totY):

    # selfobjXtasksPy => Cythonization:
    cdef int mt = len(selfobjXtasksPy)
    cdef int * selfobjXtasks = <int *>malloc(mt*cython.sizeof(int))

    if selfobjXtasks is NULL:
        raise MemoryError()
    cdef int i,j
    for i in xrange(mt):
        selfobjXtasks[i] = selfobjXtasksPy[i] 
    
    cdef np.ndarray[FLOAT_t, ndim=1] w = np.zeros((Xshape1), dtype=FFLOAT)
    cdef float *wdata = <float *> w.data
    cdef np.ndarray[FLOAT_t, ndim=1] w_accum = np.zeros((Xshape1), dtype=FFLOAT)
    cdef float *waccumdata = <float *> w_accum.data

    if mt == 0:
        print 'ERROR! Task',t,'with zero examples in the training part'
        return w # TODO: Check this
    
    # Support variables:
    cdef np.ndarray[FLOAT_t, ndim=1] v = np.zeros((Xshape1), dtype=FFLOAT)
    cdef float *vdata = <float *>v.data

    # Note: to training, task is sent zero based so no correction is needed
    cdef np.ndarray[FLOAT_t, ndim=1] Z = totZ[t*Xshape1 : (t+1)*Xshape1]
    cdef float * Zdata = <float *>Z.data
    cdef np.ndarray[FLOAT_t, ndim=1] Y = totY[t*Xshape1 : (t+1)*Xshape1]
    cdef float * Ydata = <float *>Y.data

    cdef float lam1 = selfobjlam1
    cdef float actual_prec = selfobjprec + 1.0
    cdef float lam = (selfobjro1+selfobjeta)/mt
    cdef float eta = selfobjeta

    cdef int step = 0
    cdef int samples = 0

    cdef int rand_int
    cdef int random_value
    cdef float norm2
    cdef float prec_ro
    cdef float delta
    cdef float wtmp_correction
    cdef long y
    cdef float mm

    cdef float dprodVal = 0.0
    cdef maxbag = 0
    if mt < 100:
        maxbag = mt
    else:
        maxbag = 100

    cdef float imaxbag = 1.0 / maxbag

    while (samples <= selfobjmaxSGDiter*mt and actual_prec >= selfobjprec):
        #print 'Step',step+1
        
        samples += maxbag
        step += 1
        delta = 1.0 / (lam * step)
    
        # Calculate direction
        for i in xrange(Xshape1):
            # vdata[i] = (1.0 / mt) * ( (lam/2.0) * wdata[i] + Ydata[i] - eta*Zdata[i]) # Wrong?
            vdata[i] = (lam * wdata[i]) + (1.0 / mt) * (Ydata[i] - eta*Zdata[i])  

        # Use a bunch of examples
        for j in xrange(maxbag):
            rand_int = int(rand()/float(RAND_MAX) * (mt-1))
            random_value = int(selfobjXtasks[rand_int])  # TODO: random choise of a pattern => USE FIXED SEED !
            y = labels[random_value]
            
            # Do dot product in sparse way!
            # Note columns in a row do not come in order necessarly
            dprodVal = 0.0
    
            for i in xrange(Xindptr[random_value],Xindptr[random_value+1]): # Ignore task
                if Xindices[i] > 0: # Ignore task
                    dprodVal += wdata[Xindices[i]]*Xdata[i]
    
            if (1.0 - y * dprodVal) > 0:
                for i in xrange(Xindptr[random_value],Xindptr[random_value+1]): 
                    if Xindices[i] > 0: # Ignore task
                        vdata[Xindices[i]] -= imaxbag * (Xdata[i] * y) # removed: (1.0 / mt) *
        
        # Follow direction
        norm2 = 0.0
        for i in xrange(1,Xshape1): # Ignore task dimension
            wdata[i] -= delta * vdata[i]
            norm2 += wdata[i]*wdata[i]
         
        norm2 = sqrt(norm2)

        if norm2 > 0:
            wtmp_correction = sqrt(T * M / (2 * lam1)) / norm2
        else:
            wtmp_correction = 0.0
        
        # min([1,wtmp_correction])
        if wtmp_correction < 1:
            mm = wtmp_correction
        else:
            mm = 1

        # Scale into the feasible region and accumulate average w
        for i in xrange(Xshape1):
            wdata[i] *= mm                                              # Scale
            waccumdata[i] = ((step-1)*waccumdata[i] + wdata[i])/step    # Accumulate the average

        prec_ro = (1.0 / mt*mt) * (selfobjR + lam * selfobjbeta + selfobjeta*selfobjeta * selfobjbeta * (2.0 * selfobjouter_step + 1.0)) # TOOBS: evaluate correctly the new actual precision
        actual_prec = prec_ro / (2.0 * lam * step)
    if selfobjverbose:
        print 'Precision bound for SGD of w_',t,' variable:',actual_prec,'|with limit:',selfobjprec

    # FREE THE MEMORY
    free(selfobjXtasks)

    return w_accum

@cython.boundscheck(False)
def CfindZ(float M,
          float T,
          float selfobjlam1,
          float selfobjro1,
          float selfobjro2,
          float selfobjprec,
          int selfobjmaxSGDiter,
          float selfobjeta,
          float selfobjbeta,
          int selfobjouter_step,
          bint selfobjverbose, #boolean as int
          np.ndarray[float] w,
          np.ndarray[float] y,
          np.ndarray[float] z):

    cdef int Tint = int(T)
    cdef int wshape0 = w.shape[0]
    cdef int feats = int(float(wshape0) / T)

    cdef float * zdata = <float *> z.data
    cdef float * ydata = <float *> y.data
    cdef float * wdata = <float *> w.data

    #cdef np.ndarray[DOUBLE_t, ndim=1] tmp = np.zeros((feats), dtype=DDOUBLE)
    #cdef double * tmpdata = <double *> tmp.data
    global tmpdata

    cdef float d02 = (2.0 * selfobjro2) / ((selfobjeta + selfobjro1) * (selfobjeta + selfobjro1 + 2.0 * selfobjro2) * T)
    cdef float dm2_d02 = 1.0 / (selfobjeta + selfobjro1 + 2 * selfobjro2)

    cdef float eta = selfobjeta
    cdef float ss  = 0

    for i in xrange(feats):
        tmpdata[i] = 0

    # The first block of kdata
    for i in xrange(wshape0):
        ss = ydata[i] + eta*wdata[i]
        zdata[i] = dm2_d02*ss
        tmpdata[i % feats] += ss*d02

    # Copy the first block to the others and diagonal correction
    for i in xrange(wshape0):
        zdata[i] += tmpdata[i % feats]
        
    return
    
@cython.boundscheck(False)
def CdecisionFunction(np.ndarray[cython.int, ndim=1] Xindices,
                      np.ndarray[cython.int, ndim=1] Xindptr,
                      np.ndarray[cython.double, ndim=1] Xdata,
                      int Xshape0,
                      int Xshape1,
                      np.ndarray[cython.float, ndim=1] W,
                      int T,
                      int Wshape1):
    cdef np.ndarray[DOUBLE_t, ndim=1] decisions = np.zeros((Xshape0), dtype=DDOUBLE)
    cdef int i
    cdef int k
    cdef int j
    cdef int t
    cdef int numPatterns = Xshape0
    cdef float * Wdata = <float *> W.data

    for i in range(numPatterns):
        
        decisions[i] = 0.0
        t = 100000000
        
        # Get the task, it is the first value in the row
        # Note columns in a row do not come in order necessarly
        for k in xrange(Xindptr[i],Xindptr[i+1]):
            if Xindices[k] == 0:
                t = <int>Xdata[k]
                break

        if t > T: # Not in our list of tasks
            print 'not in our tasks', t
            decisions[i] = 0.0
            continue

        # Note, ignore task value
        # Note, tasks are one based to be able to use sparse, correction needed
        for j in xrange(Xindptr[i],Xindptr[i+1]):
            if Xindices[j] > 0: # Ignore task
                decisions[i] += Wdata[Wshape1*(t-1) + Xindices[j]]*Xdata[j]

    return decisions
