from sklearn.preprocessing import scale
from SVM import TrainMyClassifierSVM
from RVC import TrainMyClassifierRVM
from GPR import TrainMyClassifierGPR


def TrainMyClassifier(XEstimate, XValidate, Parameters, ClassLabels):
    '''
        This function trains the corresponding classifier based on the choice
        The Parameters argument must be a string in the set {'RVM', 'SVM', 'GPR'},
        indicating which classification algorithm to train.
    '''
    # standardize the estimate and validate data
    XEstimateScaled = scale(XEstimate)
    XValidateScaled = scale(XValidate)

    if(Parameters == 'RVM'):
        EstParameters = TrainMyClassifierRVM(XEstimateScaled, ClassLabels)
    elif(Parameters == 'SVM'):
        EstParameters = TrainMyClassifierSVM(XEstimateScaled, ClassLabels)
    elif(Parameters == 'GPR'):
        EstParameters = TrainMyClassifierGPR(XEstimateScaled, ClassLabels)
    else:
        raise ValueError(
            'Wrong parameters value. The values can either be RVM, SVM or GPR')

    YValidate = EstParameters.predict(XValidateScaled)
    return YValidate, EstParameters
