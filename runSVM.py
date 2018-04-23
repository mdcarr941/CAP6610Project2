import scipy.io as sio
import numpy as np
from sklearn.model_selection import KFold
from utility_functions import loaddata
from sklearn.cross_validation import train_test_split
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import SVM
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
    estParam = SVM.TrainMyClassifierSVM(X_estimate, y_estimate, param)
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