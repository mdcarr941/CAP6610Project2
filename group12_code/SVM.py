#import scipy.io as sio
import numpy as np
#from sklearn.model_selection import KFold
#from utility_functions import loaddata
#from sklearn.cross_validation import train_test_split
import sklearn.svm as svm
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report


def TrainMyClassifierSVM(X_train, y_train, ifprint = False, **kwargs):
    clf = svm.SVC(kernel="linear", C=2.0, decision_function_shape = 'ovo', **kwargs).fit(X_train, y_train)
    # hyperParam = clf.get_params()
    # #create a dictionary for estimated parameters
    # estParam = {}
    # estParam["support_"] = clf.support_
    # estParam["_dual_coef_"] = clf._dual_coef_
    # estParam["n_support_"] = clf.n_support_
    # estParam["support_vectors_"] = clf.support_vectors_
    # estParam["_intercept_"] = clf._intercept_
    # estParam["_sparse"] = clf._sparse
    # estParam["shape_fit_"] = clf.shape_fit_
    # estParam["probA_"] = clf.probA_
    # estParam["probB_"] = clf.probB_
    # estParam["_gamma"] = clf._gamma
    # estParam["classes_"] = clf.classes_
    #return [estParam, hyperParam, y_predict]
    if ifprint:
        print "the number of support vectors is: ", clf.n_support_, len(clf.support_)
    return clf

def TestMyClassifierSVM(XTest, EstParameters):
    # estParam = EstParameters[0]
    # hyperParam = EstParameters[1]
    # clf2 = svm.SVC()
    # clf2.set_params(**hyperParam)
    # clf2.support_ = estParam["support_"]
    # clf2._dual_coef_  = estParam["_dual_coef_"]
    # clf2.n_support_   = estParam["n_support_"]
    # clf2.support_vectors_    = estParam["support_vectors_"]
    # clf2._intercept_    = estParam["_intercept_"]
    # clf2._sparse = estParam["_sparse"]
    # clf2.shape_fit_ = estParam["shape_fit_"]
    # clf2.probA_ = estParam["probA_"]
    # clf2.probB_ = estParam["probB_"]
    # clf2._gamma = estParam["_gamma"]
    # clf2.classes_ = estParam["classes_"]
    #clf2 = EstParameters[0]
    clf2 = EstParameters
    y_predict = clf2.predict(XTest)
    return y_predict