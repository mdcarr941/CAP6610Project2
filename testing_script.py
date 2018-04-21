#!/usr/bin/env python2
# Script used to test code written for this project.

from utility_functions import *

def test_make_confusion_matrix(targets, estimates, expectedA, doprint=False):
    '''
    Create a confusion matrix using targets and estimates and assert that it
    equals expectedA. If doprint is true then the computed confusion matrix A,
    expectedA, and A == expectedA are printed.
    '''
    A = make_confusion_matrix(targets, estimates)
    assert_col_sum_one(A)
    assert(A == expectedA).all()
    if doprint: 
        print A
        print expectedA
        print (A == expectedA)

def assert_col_sum_one(A):
    ''' Check that the columns of a confusion matrix sum to one. '''
    assert( (A.sum(axis=1) == np.ones(A.shape[1])).all() )
    

targets =   [1, 1, 0, 1]
estimates = [1, 0, 0, 1]
expectedA = np.array([
    [1.  , 0.  ],
    [1./3, 2./3]
])
test_make_confusion_matrix(targets, estimates, expectedA)

targets =   [1, 1, 0, 1, 0]
estimates = [1, 0, 0, 1, 1]
expectedA = np.array([
    [1. / 2, 1. / 2],
    [1. / 3, 2. / 3]
])
test_make_confusion_matrix(targets, estimates, expectedA)

print 'All tests passed.'
