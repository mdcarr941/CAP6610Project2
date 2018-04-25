import numpy
#import GPy
import scipy
import utility_functions
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
import testing_script


(trainingData, labelsVector) = utility_functions.loaddata(doflatten=False)

# Gaussian Process Regressor
# gpr = GPR()
# gpr.fit(trainingData[4998:5003], labelsVector[4998:5003])
# mean, cov = gpr.predict(trainingData[4998:5003], return_cov=True)
# mean = utility_functions.flatten_targets(mean)
# print(mean)

# Gaussian Process Classifier
labelsVector = utility_functions.flatten_targets(labelsVector)


def TrainMyClassifierGPR(X_train, y_train, **kwargs):
    gpc = GPC(kernel=1.0 * RBF(length_scale=1.0),
              multi_class='one_vs_one', optimizer=None)
    gpc.fit(X_train, y_train)
    return gpc


def TestMyClassifierGPR(XTest, EstParameters):
    y_predict = []
    clf = EstParameters
    probs = clf.predict_proba(XTest)
    for probIter in probs:
        if(max(probIter) == 1/numpy.size(probIter)):
            y_predict.append(-1)
        else:
            index = numpy.argmax(probIter, axis=None)
            y_predict.append(index)
    return y_predict

    # kernelFunction = GPy.kern.RBF(input_dim=DIMENSION)  # Can change kernel
    #model = GPy.models.GPRegression(trainingData[0:5], labelsVector[0:5], kernelFunction)
    # model.optimize(max_iters=1)
    # model.optimize_restarts(num_restarts=5)
    #(mean, var) = model.predict(trainingData[6:10])
    # GPy.models.OneVsAllClassification()
    # print(mean)
    # model.optimize_restarts(num_restarts=10)
    # display(model)
    # figure = model.plot()
    # GPy.plotting.show(figure, fileName='GaussianProcessRegression')

    # model.optimize_restarts()
    # model.plot()
