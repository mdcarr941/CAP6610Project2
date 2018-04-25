from SVM import TestMyClassifierSVM
from RVC import TestMyClassifierRVM
from GPR import TestMyClassifierGPR

def TestMyClassifier(XTest, Parameters, Classifier):
    '''
    Run a trained classifer on a given set of data and return the resulting class labels.
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
