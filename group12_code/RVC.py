#from skrvm import RVC
from rvm import RVC
from sklearn.multiclass import OneVsOneClassifier

def TrainMyClassifierRVM(X_train, y_train, ifprint = False, **kwargs):
    #classifier = OneVsOneClassifier(RVC(n_iter=1, **kwargs))
    #classifier1 = RVC(n_iter=1, verbose=True, **kwargs)
    #classifier1.fit(X_train, y_train)
    classifier = OneVsOneClassifier(RVC(n_iter=1, **kwargs))
    classifier.fit(X_train, y_train)
    if ifprint:
        #print "the number of support vectors is: ", len(classifier.estimators_)
        numVct = 0
        for clss in classifier.estimators_:
            numVct += clss.relevance_.shape[0]
            print clss.relevance_.shape[0]
        print "the number of support vectors is: ", numVct
    return classifier

def TestMyClassifierRVM(XTest, EstParameters):
    clf2 = EstParameters
    y_predict = clf2.predict(XTest)
    return y_predict