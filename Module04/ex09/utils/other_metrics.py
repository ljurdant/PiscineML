import numpy as np
import sys

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        correct_count = np.count_nonzero(y == y_hat)
        return correct_count / y_hat.shape[0]
    except:
        return None


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        tp = np.logical_and(y_hat == pos_label, y == y_hat )
        fp = np.logical_and(y_hat == pos_label, y != y_hat)
        tp_count = np.count_nonzero(tp)
        fp_count = np.count_nonzero(fp)
        if (tp_count + fp_count) == 0:
            return 0
        return tp_count / (tp_count + fp_count)
    except Exception as error:
        print(f"{error} in function {__name__}", file=sys.stderr)
        return None

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        tp = np.logical_and(y_hat == pos_label, y == y_hat)
        fn = np.logical_and(y_hat != pos_label, y != y_hat)
        tp_count = np.count_nonzero(tp)
        fn_count = np.count_nonzero(fn)
        if (tp_count + fn_count) == 0:
            return 0
        return tp_count / (tp_count + fn_count)
    except Exception as error:
        print(f"{error} in function {__name__}", file=sys.stderr)
        return None
    

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """

    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    
    try:
        if precision + recall == 0:
            return 0
        return (2*precision*recall) / (precision + recall)
    except Exception as error:
        print(f"{error} in function {__name__}", file=sys.stderr)
        return None