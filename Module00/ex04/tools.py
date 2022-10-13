import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """
    if isinstance(x, np.ndarray):
        if len(x) and len(x.shape) < 3:
            X = np.column_stack((np.full(x.shape[0], 1.0), x))
            return X