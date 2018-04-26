from utility_functions import *

def MyConfusionMatrix(Y, ClassLabels):
    '''
    Compute a confusion matrix for an estimate vector Y and a target vector ClassLabels.
    Arguments:
        Y - ndarray of shape (N,), the estimated class labels
        ClassLabels - ndarray of shape (N,), the actual class labels
    Returns
        A - ndarray, the confusion matrix
        AvgAccuracy - float, the average accuracy of the estimates
    '''

    A = make_confusion_matrix(ClassLabels, Y)
    print_confusion_matrix(A)
    AvgAccuracy = average_accuracy(ClassLabels, Y)
    return A, AvgAccuracy
