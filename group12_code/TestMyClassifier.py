from utility_functions import *

def TestMyClassifier(XTest, Parameters, EstParameters):
    '''
    Arguments:
        XTest - A matrix of feature vectors, where each row is a sample vector.
        Parameters - A dictionary of parameters. Algorithm dependant.
        EstParameters - A dictionary of the fitted model parameters. 
    Returns:
        Ytest - Class labels for each row of XTest.
    '''
