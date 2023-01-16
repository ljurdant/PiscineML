import sys, os
dirname = os.path.dirname(os.path.abspath(__file__))[:-4]+"utils"
sys.path.append(dirname)
from TinyStatistician import TinyStatistician

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn't raise any Exception.
    """
    ts = TinyStatistician()
    if ts.mean(x) and ts.std(x):
        return (x - ts.mean(x)) / ts.std(x)