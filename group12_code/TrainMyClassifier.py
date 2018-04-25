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

    try:
        algorithm = Parameters.pop('algorithm')
    except KeyError:
        raise ValueError(
            'The Parameters dictionary must contain the key "algorithm".'
        )

    if(algorithm == 'RVM'):
        EstParameters = TrainMyClassifierRVM(XEstimateScaled, ClassLabels, **Parameters)
    elif(algorithm == 'SVM'):
        EstParameters = TrainMyClassifierSVM(XEstimateScaled, ClassLabels, **Parameters)
    elif(algorithm == 'GPR'):
        EstParameters = TrainMyClassifierGPR(XEstimateScaled, ClassLabels, **Parameters)
    else:
        raise ValueError(
            'Wrong parameters value. The values can either be RVM, SVM or GPR'
        )

    # Return the algorithm entry to the dictionary (because side effects are bad).
    Parameters['algorithm'] = algorithm

    YValidate = EstParameters.predict(XValidateScaled)
    return YValidate, EstParameters
