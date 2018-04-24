from utility_functions import *

def TestMyClassifier(XTest, Parameters, Classifier):
    '''
    Arguments:
        XTest - A matrix of feature vectors, where each row is a sample vector.
        Parameters - A dictionary of parameters. Algorithm dependant.
        Classifier - A classifier object which has already been fitted to the data.
    Returns:
        Ytest - Class labels for each row of XTest.
    '''
    
    return Classifier.predict(XTest) 
