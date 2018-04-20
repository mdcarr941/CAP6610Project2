#!/usr/bin/env python2
from scipy.io import loadmat 
import numpy as np

FEATVECFILE = 'Proj2FeatVecsSet1.mat'
TARGETFILE = 'Proj2TargetOutputsSet1.mat'

def flatten_targets(y):
    ''' The targets which are vectors to scalars. '''
    ymod = np.ndarray(y.shape[0], dtype=np.int32)
    for k in range(y.shape[0]):
        ymod[k] = y[k].argmax()
    return ymod

def loaddata():
    ''' Load the mat files containing feature vecs and targets. '''
    mat = loadmat(FEATVECFILE)
    X = mat['Proj2FeatVecsSet1']
    mat = loadmat(TARGETFILE)
    y = mat['Proj2TargetOutputsSet1']
    return X, flatten_targets(y) 

def argfirst(vec, entry):
    ''' Find the first index in vec of the value entry. ''' 
    for k in range(vec.shape[0]):
        if entry == vec[k]:
            return k
    return None

