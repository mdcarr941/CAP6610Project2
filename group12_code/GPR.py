from sklearn.gaussian_process import GaussianProcessRegressor as GPR


def TrainMyClassifierGPR(X_train, y_train, ifprint = False, **kwargs):
    #classifier = OneVsOneClassifier(RVC(n_iter=1, **kwargs))
    #classifier1 = RVC(n_iter=1, verbose=True, **kwargs)
    #classifier1.fit(X_train, y_train)
    classifier = GPR()
    return classifier

def TestMyClassifierGPR(XTest, EstParameters):
    clf2 = EstParameters
    y_predict = clf2.predict(XTest)
    return y_predict
########################################################################################################
# DIMENSION = 60

# (trainingData, labelsVector) = utility_functions.loaddata()
# GPR.fit(trainingData[0:5], labelsVector[0:5])
# (mean, std, cov) = GPR.predict(trainingData[6:10])
# print(mean)


#kernelFunction = GPy.kern.RBF(input_dim=DIMENSION)  # Can change kernel
#model = GPy.models.GPRegression(trainingData[0:5], labelsVector[0:5], kernelFunction)
#model.optimize(max_iters=1)
# model.optimize_restarts(num_restarts=5)
#(mean, var) = model.predict(trainingData[6:10])
#GPy.models.OneVsAllClassification()
#print(mean)
# model.optimize_restarts(num_restarts=10)
# display(model)
# figure = model.plot()
# GPy.plotting.show(figure, fileName='GaussianProcessRegression')

# model.optimize_restarts()
# model.plot()
