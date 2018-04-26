from sklearn.preprocessing import scale
from SVM import TrainMyClassifierSVM
from RVC import TrainMyClassifierRVM
from GPR import TrainMyClassifierGPR


def TrainMyClassifier(XEstimate, XValidate, Parameters, ClassLabels):
    '''
    Train a classifier on XEstimate using class labels ClassLabels
    and test it using XValidate.
    Arguments:
        XEstimate - ndarray of shape (N, M), the data to fit the model to
        XValidate - ndarray of shape (N, M), the data to validate the model on
        Parameters - dictionary, algorithm specific parameters. Must contain key 
            'algorithm' with value 'RVM', 'SVM', or 'GPR'
        ClassLabels - ndarray of shape (N,), the class labels
            of the samples in XEstimate
    Returns:
        YValidate - ndarray of shape (N,), the estimated class labels for XValidate
        EstParameters - kn sklearn classifier object fitted to the data
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
