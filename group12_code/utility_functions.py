# Various utility functions.

from scipy.io import loadmat 
import numpy as np
import numpy.random

def flatten_targets(y):
    ''' The targets which are vectors to scalars. '''
    ymod = np.ndarray(len(y), dtype=np.int32)
    for k in range(len(y)):
        ymod[k] = y[k].argmax()
    return ymod

def loaddata(
    featvecfile='Proj2FeatVecsSet1.mat',
    targetfile='Proj2TargetOutputsSet1.mat',
    doflatten=True
):
    '''
    Load the mat files containing feature vecs and targets. If the optional argument
    doflatten is true then the targets will be flattened into a vector.
    '''
    mat = loadmat(featvecfile)
    X = mat['Proj2FeatVecsSet1']
    mat = loadmat(targetfile)
    y = mat['Proj2TargetOutputsSet1']
    if doflatten:
        y = flatten_targets(y)
    return X, y 

def argfirst(vec, entry):
    ''' Find the first index in vec of the value entry. ''' 
    for k in range(len(vec)):
        if entry == vec[k]:
            return k
    return None

def make_confusion_matrix(targets, estimates):
    '''
    Create a confusion matrix. targets is a vector of target class labels and
    estimates are the class labels assigned by the classifier. It is expected
    that targets.shape = (N,) = estimates.shape, where N >= 0, and the values
    in estimates are contained in the set of values in targets.
    If this is not the case then this function may return unexpected values.
    '''
    class_lbls = np.unique(targets)

    estimates_idx = {}
    for m in class_lbls:
        estimates_idx[m] = (estimates == m)

    N = len(class_lbls)
    A = np.ndarray((N, N), dtype=np.float)
    for m in class_lbls:
        targets_idx_m = (targets == m)
        for n in class_lbls:
            A[m][n] = (estimates_idx[n] & targets_idx_m).sum() 
        A[m] /= targets_idx_m.sum()

    return A

def print_confusion_matrix(A, precision=3):
    '''
    Print a (confusion) matrix to the console. The parameter precision is the number
    of decimal places to include in the output.
    '''
    format_string = '{0:' + str(precision + 1) + '.' + str(precision) + 'f}'
    for row in A:
        print '[',
        for k in range(len(row) - 1):
            entry = row[k]
            print format_string.format(entry) + ',',
        print format_string.format(row[-1]) + ' ]'

def average_accuracy(targets, estimates):
    try:
        return (targets == estimates).sum() / len(targets)
    except ZeroDivisionError:
        if len(estimates) == 0:
            return 1
        return 0

def make_random_indices(length, total):
    indices = np.arange(total)
    np.random.shuffle(indices)
    return indices[:length]
