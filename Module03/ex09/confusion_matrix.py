import numpy as np
import pandas as pd

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
            This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
            instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """

    if labels == None:
        labels = np.unique(np.concatenate((y_true, y_hat)))
    matrix = []
    try:
        for label in labels:
            row = []
            for other in labels:
                row.append(np.count_nonzero(np.logical_and(y_true == label, y_hat == other)))
            matrix.append(row)
  
   
        if df_option:
            return pd.DataFrame(matrix, labels, labels)
        else:
            return np.array(matrix)
    except: 
        return None