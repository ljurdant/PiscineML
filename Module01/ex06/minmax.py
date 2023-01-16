import sys, os
dirname = os.path.dirname(os.path.abspath(__file__))[:-4]+"utils"
sys.path.append(dirname)
import numpy as np

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn't raise any Exception.
    """    
    if isinstance(x, np.ndarray):
        return (x - min(x)) / (max(x) - min(x))