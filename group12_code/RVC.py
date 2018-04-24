from fast_rvm import RVC
from sklearn.datasets import load_iris
from scipy.io import loadmat
from utility_functions import *
from random import randint,sample
import numpy as np
from sklearn.metrics import accuracy_score

FEATVECFILE = 'Proj2FeatVecsSet1.mat'
TARGETFILE = 'Proj2TargetOutputsSet1.mat'

print 'loading mat files'
# load the .mat files
mat = loadmat(FEATVECFILE)
X = np.array(mat['Proj2FeatVecsSet1'])
mat = loadmat(TARGETFILE)
y = np.array(flatten_targets(mat['Proj2TargetOutputsSet1']))

dataMatrix = np.c_[X,y]
# print X.shape
# print y.shape
# print dataMatrix.shape
# classifier from rvm.py
classifier = RVC()
#iris =  load_iris()
np.random.shuffle(dataMatrix)
trainData, testData = dataMatrix[:100,:], dataMatrix[100:110,:]
# print trainData.shape
# print testData.shape

print 'creating training data'
# sample and labels for training
temp = trainData[:,trainData.shape[1]-1].tolist()
labels = [int(l) for l in temp]
trainSamples = np.delete(trainData, np.s_[-1:], axis=1)

print 'creating testing data'
# sample and labels for testing
temp = testData[:,testData.shape[1]-1].tolist()
y_true = [int(l) for l in temp]
testSamples = np.delete(testData, np.s_[-1:], axis=1)
# print samples.shape

print 'training'
#fit
classifier.fit(trainSamples,labels)
#print("Relevance Vectors: {}".format(classifier.relevance_.shape[0]))
print classifier.get_params();

print 'testing'
#predict
y_pred = classifier.predict(testSamples) 

print 'confusion matrix and accuracy'
#confusion matrix and average accuracy)
confMatrix = make_confusion_matrix(y_true,y_pred)
averageAcc = accuracy_score(y_true,y_pred)
print confMatrix
print averageAcc
