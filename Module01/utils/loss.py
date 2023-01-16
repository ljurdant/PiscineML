import numpy as np

def loss_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        y_hat = np.reshape(y_hat, (max(y_hat.shape), 1))
        y = np.reshape(y, (max(y.shape), 1))
    except:
        return None
    else:
        if not y.shape == y_hat.shape:
            return None
        else:
            return (y_hat - y)**2
			
def loss_(y, y_hat):
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    J_elem = loss_elem_(y, y_hat)
    if not isinstance(J_elem, type(None)):
        return np.mean(J_elem) / 2