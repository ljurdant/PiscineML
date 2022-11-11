import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_
from loss import loss_, loss_elem_

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    try:
        np.reshape(x, (max(x.shape), 1))
        np.reshape(y, (max(y.shape), 1))
        np.reshape(theta, (theta.shape[0], 1))
    except:
        return None
    else:
        y_hat = predict_(x, theta)
        plt.plot(x, y, "o",x, y_hat)
        
        #Plotting difference lines
        for i in range(len(x)):
            plt.plot([x[i],x[i]],[y[i],y_hat[i]], color="red", linestyle="dashed")

        #Print loss
        loss = loss_(y, y_hat)
        plt.title("Cost: "+str(loss))
        plt.show()