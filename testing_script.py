#!/usr/bin/env python2
# Script used to test code written for this project.

from group12_code import *
import traceback
import sys

class TestRunner:
    ''' This class keeps a list of test cases and runs them all when requested. '''
    tests = []
    failed = []

    def add(self, test):
        ''' Add a test to the list. '''
        self.tests.append(test)
        return test

    def run(self, names=None):
        ''' Run all tests in the list. '''
        print 'Running tests.'
        for test in self.tests:
            if names != None and not test.__name__ in names:
                continue
            print
            try:
                test()
                print 'Test passed:', test.__name__
            except Exception as e:
                print 'Exception in:', test.__name__
                traceback.print_exc(file=sys.stdout)
                self.failed.append(test)
        print
        print self

    def __str__(self):
        output = 'All tests passed.'
        if len(self.failed) > 0:
            output = 'Failed tests: '
            for k in range(len(self.failed) - 1):
                output += self.failed[k].__name__ + ', '
            output += self.failed[-1].__name__
        return output
    
              

testrunner = TestRunner()

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

@testrunner.add
def confusion_matrix_test_case1():
    targets =   np.array([1, 1, 0, 1])
    estimates = np.array([1, 0, 0, 1])
    expectedA = np.array([
        [1.  , 0.  ],
        [1./3, 2./3]
    ])
    test_make_confusion_matrix(targets, estimates, expectedA)

@testrunner.add
def confusion_matrix_test_case2():
    targets =   np.array([1, 1, 0, 1, 0])
    estimates = np.array([1, 0, 0, 1, 1])
    expectedA = np.array([
        [1. / 2, 1. / 2],
        [1. / 3, 2. / 3]
    ])
    test_make_confusion_matrix(targets, estimates, expectedA)

@testrunner.add
def test_MyConfusionMatrix():
    targets =   np.array([1, 1, 0, 1, 0])
    estimates = np.array([1, 0, 0, 1, 1])
    MyConfusionMatrix(estimates, targets)

@testrunner.add
def test_SVC():
    import scipy.io as sio
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.cross_validation import train_test_split
    import sklearn.svm as svm
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import random

    # features = sio.loadmat("Proj2FeatVecsSet1.mat").get("Proj2FeatVecsSet1")
    # targetOutput = sio.loadmat("Proj2TargetOutputsSet1.mat").get("Proj2TargetOutputsSet1")
    #read in data
    [features, targetOutput] = loaddata()
    # print features, features.shape
    # print targetOutput, targetOutput.shape
    index = np.arange(features.shape[0])
    #index = np.arange(10)
    random.shuffle(index)
    #print index, features.shape[0], index[0:3]
    # print features[index[0:3]]
    # print features[index[0:3]].shape
    index_train = index[0:20000]
    index_test = index[20000:]
    features_train = features[index_train]
    features_test = features[index_test]
    targetOutput_train = targetOutput[index_train]
    targetOutput_test = targetOutput[index_test]


    #X_estimate, X_validate, y_estimate, y_validate = train_test_split(features_train, 
    #    targetOutput_train, test_size=0.2, random_state=42)
    param = ["linear", 2.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=17)
    score = 0.0
    score1 = 0.0
    for train_index, test_index in kf.split(features_train):
        X_estimate, X_validate = features_train[train_index], features_train[test_index]
        y_estimate, y_validate = targetOutput_train[train_index], targetOutput_train[test_index]
        estParam = SVM.TrainMyClassifierSVM(X_estimate, y_estimate)
        score1 = estParam.score(X_validate,y_validate)
        score += score1
        print score1

    print "average score: ", score/5.0

    y_predict_clf2 = SVM.TestMyClassifierSVM(features_test, estParam)

    confMat = confusion_matrix(targetOutput_test, y_predict_clf2)
    print confMat
    print(classification_report(targetOutput_test, y_predict_clf2))
    print estParam.score(features_test,targetOutput_test)
    #dec = estParam.decision_function(features_test)
    #print dec[:5]
    #print y_predict_clf2[:5]


    # clf = svm.SVC(kernel='linear', C=2).fit(X_train, y_train)
    # hyperParam = clf.get_params(deep=True)
    # clf2 = svm.SVC()
    # clf2.set_params(**hyperParam)
    # clf2.support_ = clf.support_
    # clf2._dual_coef_  = clf.dual_coef_ 
    # clf2.n_support_   = clf.n_support_ 
    # clf2.support_vectors_    = clf.support_vectors_ 
    # clf2._intercept_    = clf.intercept_ 
    # clf2._sparse = clf._sparse
    # clf2.shape_fit_ = clf.shape_fit_
    # clf2.probA_ = clf.probA_
    # clf2.probB_ = clf.probB_
    # clf2._gamma = clf._gamma
    # clf2.classes_ = clf.classes_


    #print(classification_report(y_test, y_predict))
    #print clf.score(X_test,y_test)


    # X = np.array([[1, 2], [3, 4], [3, 5], [3, 7], [2, 6], [5,8], [1, 10], [7,7], [6,3], [4,10]])
    # y = np.array([1, 2, 1, 1, 2, 2, 1, 2, 2, 1])
    # kf = KFold(n_splits=5, shuffle=True, random_state=0)
    # for train_index, test_index in kf.split(X):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     #print("Train data: ", X_train, "Target output:", y_train)
    #     #print("Test data: ", X_test, "Test target output:", y_test)

@testrunner.add
def test_train_svm():
    X, y = loaddata()
    validate_idx = make_random_indices(200, len(y))
    estimate_idx = make_random_indices(1000, len(y))
    return TrainMyClassifier(
        X[estimate_idx], X[validate_idx],
        'SVM', y[estimate_idx]
    )

@testrunner.add
def test_train_rvm():
    X, y = loaddata()
    validate_idx = make_random_indices(200, len(y))
    estimate_idx = make_random_indices(1000, len(y))
    return TrainMyClassifier(
        X[estimate_idx], X[validate_idx],
        'RVM', y[estimate_idx]
    )

@testrunner.add
def test_cross_validation():
    X, y = loaddata()
    indices = make_random_indices(1000, len(y))
    return MyCrossValidate(X[indices], 5, 'SVM', y[indices])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        testrunner.run(names=sys.argv)
    else:
        testrunner.run()
