import numpy
#import GPy
import scipy
import utility_functions
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
import testing_script

DIMENSION = 60

(trainingData, labelsVector) = utility_functions.loaddata(doflatten=False)

# Gaussian Process Regressor
gpr = GPR()
gpr.fit(trainingData[4998:5003], labelsVector[4998:5003])
mean, cov = gpr.predict(trainingData[4998:5003], return_cov=True)
mean = utility_functions.flatten_targets(mean)
print(mean)

# Gaussian Process Classifier
labelsVector = utility_functions.flatten_targets(labelsVector)
gpc = GPC(kernel=1.0 * RBF(length_scale=1.0), multi_class='one_vs_one')
gpc.fit(trainingData[:20000], labelsVector[:20000])
probs = gpc.predict_proba(trainingData[5003:5009])
print(probs)

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
