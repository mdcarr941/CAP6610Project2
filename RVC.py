from skrvm import RVC
from sklearn.datasets import load_iris
from scipy.io import loadmat
from utility_functions import *
from random import randint,sample
import numpy as np
from sklearn.metrics import accuracy_score
import random
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsOneClassifier

def TrainMyClassifierRVM(X_train, y_train):
    classifier = OneVsOneClassifier(RVC(n_iter=1))
    classifier.fit(X_train, y_train)
    return classifier

def TestMyClassifierRVM(XTest, EstParameters):
    clf2 = EstParameters
    y_predict = clf2.predict(XTest)
    return y_predict