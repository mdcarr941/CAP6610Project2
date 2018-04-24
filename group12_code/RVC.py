from skrvm import RVC
from sklearn.multiclass import OneVsOneClassifier

def TrainMyClassifierRVM(X_train, y_train):
    classifier = OneVsOneClassifier(RVC(n_iter=1))
    classifier.fit(X_train, y_train)
    return classifier

def TestMyClassifierRVM(XTest, EstParameters):
    clf2 = EstParameters
    y_predict = clf2.predict(XTest)
    return y_predict