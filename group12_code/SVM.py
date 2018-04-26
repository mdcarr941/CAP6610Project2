import numpy as np
import sklearn.svm as svm
from sklearn.multiclass import OneVsOneClassifier

def TrainMyClassifierSVM(X_train, y_train, ifprint = False, **kwargs):
    #train classifiers using SVM
    clf = svm.SVC(kernel="linear", C=2.0, decision_function_shape = 'ovo', probability=True, **kwargs)
    clf.fit(X_train, y_train)
    if ifprint:
        print "the number of support vectors is: ", clf.n_support_, len(clf.support_) #numVct
    return clf

def TestMyClassifierSVM(XTest, EstParameters):
    #test classifier using SVM
    y_predict = []
    clf2 = EstParameters
    probabilities = clf2.predict_proba(XTest)
    for probs in probabilities:
        if(max(probs) < 0.3): #this sample is not in any of the classes
            y_predict.append(-1)
        else:
            index = np.argmax(probs,axis=None)
            y_predict.append(index)
    return y_predict