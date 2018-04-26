from SVM import TestMyClassifierSVM
from RVC import TestMyClassifierRVM
from GPR import TestMyClassifierGPR

def TestMyClassifier(XTest, Parameters, Classifier):
    '''
    Run a trained classifer on a given set of data and return the resulting class labels.
    Arguments:
        XTest - ndarray of shape (N, M), the data to classify
        Parameters - dictionary, algorithm specific parameters. Must contain key 
            'algorithm' with value 'RVM', 'SVM', or 'GPR'
        Classifier - a fitted sklearn classifier object. 
    Returns:
        YTest - ndarray of shape (N,), the estimated class labels for XTest       
    '''

    try:
        algorithm = Parameters.pop('algorithm')
    except KeyError:
        raise ValueError(
            'The Parameters dictionary must contain the key "algorithm".'
        )
    
    if(algorithm == 'RVM'):
        YTest = TestMyClassifierRVM(XTest, Classifier, **Parameters)
    elif(algorithm == 'SVM'):
        YTest = TestMyClassifierSVM(XTest, Classifier, **Parameters)
    elif(algorithm == 'GPR'):
        YTest = TestMyClassifierGPR(XTest, Classifier, **Parameters)
    else:
        raise ValueError('Wrong parameters value. The values can either be RVM, SVM or GPR')

    # Return the algorithm entry to the dictionary (because side effects are bad).
    Parameters['algorithm'] = algorithm

    return YTest
