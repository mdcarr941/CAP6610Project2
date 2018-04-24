from utility_functions import *

def MyConfusionMatrix(Y, ClassLabels):
    '''
    Compute a confusion matrix for an estimate vector Y and a target vector ClassLabels.
    '''

    A = make_confusion_matrix(ClassLabels, Y)
    print_confusion_matrix(A)
    return A, A.mean()
