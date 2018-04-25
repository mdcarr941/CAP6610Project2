from SVM import TestMyClassifierSVM
from RVC import TestMyClassifierRVM
from GPR import TestMyClassifierGPR

def TestMyClassifier(XTest, Parameters, Classifier):
    '''
    Arguments:
        XTest - A matrix of feature vectors, where each row is a sample vector.
        Parameters - A dictionary of parameters. Algorithm dependant.
        Classifier - A classifier object which has already been fitted to the data.
    Returns:
        Ytest - Class labels for each row of XTest.
    '''
    
    if(Parameters == 'RVM'):
        YTest = TestMyClassifierRVM(XTest,Classifier)
    elif(Parameters == 'SVM'):
        YTest = TestMyClassifierSVM(XTest,Classifier)
    elif(Parameters == 'GPR'):
        YTest = TestMyClassifierGPR(XTest,Classifier)
    else:
        raise ValueError('Wrong parameters value. The values can either be RVM, SVM or GPR')

    return YTest
