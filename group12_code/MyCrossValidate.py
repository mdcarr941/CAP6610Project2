from TrainMyClassifier import TrainMyClassifier
from utility_functions import *
import numpy as np

def MyCrossValidate(XTrain, Nf, Parameters, ClassLabels):
    '''
    Divide a set of data up in Nf chunks, train on the complement of the chunk,
    and then classify the chunk.
    Arguments:
        XTrain - ndarray of shape (N, M), the dataset to use for cross validation
        Nf - integer, the number of folds
        Parameters - dictionary, algorithm specific parameters. Must contain key 
            'algorithm' with value 'RVM', 'SVM', or 'GPR'
        ClassLabels - ndarray of shape (N,), the class labels of the samples in XTrain
    Returns:
        YTrain - ndarray of shape(N,), the estimated class labels of the samples
            in XTrain       
        EstParametersList - list of length Nf, sklearn classifier objects fitted
            for each fold       
        EstConfMatrices - list of length Nf, the confusion matrices from each fold
        ConfMatrix - ndarray, the overall confusion matrix       
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
        mask = np.zeros((num_samples,), dtype=np.bool)
        mask[indices[start:end]] = np.ones((end - start,), dtype=np.bool)
        masks.append(mask)
        start = end

    # Train, test, and compute a confusion matrix for each fold.
    EstConfMatrices = []
    EstParametersList = []
    Ytrain = np.ndarray(ClassLabels.shape, dtype=ClassLabels.dtype)
    for mask in masks:
        YValidate, EstParameters = TrainMyClassifier(
            XTrain[~mask], XTrain[mask], Parameters, ClassLabels[~mask]
        )
        EstConfMatrices.append(make_confusion_matrix(ClassLabels[mask], YValidate))
        EstParametersList.append(EstParameters)
        Ytrain[mask] = YValidate

    # Compute the overall confusion matrix and return.
    ConfMatrix = make_confusion_matrix(ClassLabels, Ytrain)
    return Ytrain, EstParametersList, EstConfMatrices, ConfMatrix
