from TrainMyClassifier import *
from utility_functions import *
import numpy as np

def MyCrossValidate(XTrain, Nf, Parameters, ClassLabels):
    '''
    Perform cross validation.
    '''

    # Determine the size of each fold.
    num_samples = len(XTrain)
    quotient = num_samples // Nf
    remainder = num_samples - Nf * quotient
    fold_sizes = Nf * [quotient]
    while remainder > 0:
        fold_sizes[:-remainder] += 1
        remainder -= 1

    # Create a randomly shuffled index vector.
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Create a boolean mask for each fold.
    start = 0
    end  = 0
    masks = []
    for fold_size in fold_sizes:
        end += fold_size
        masks.append(indices == np.arange(start, end))
        start = end

    # Train, test, and compute a confusion matrix for each fold.
    EstConfMatrices = []
    EstParametersList = []
    Ytrain = np.ndarray(ClassLabels.shape, dtype=ClassLabels.dtype)
    for mask in masks:
        XEstimate = XTrain[~mask]
        XValidate = XTrain[mask]
        YValidate, EstParameters = TrainMyClassifier(XEstimate, XValidate, Parameters, ClassLabels)
        EstConfMatrices.append(make_confusion_matrix(ClassLabels[mask], YValidate))
        EstParametersList.append(EstParameters)
        Ytrain[mask] = YValidate

    # Compute the overall confusion matrix and return.
    ConfMatrix = make_confusion_matrix(ClassLabels, Ytrain)
    return Ytrain, EstParametersList, EstConfMatrices, ConfMatrix
