import numpy as np
import sys

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """


    try:
        total = np.append(x, y, axis=1)
        np.random.shuffle(total[:])
        y_total = total[:,-1]
        x_total = total[:,:-1]
        split_index = int(proportion*x_total.shape[0])
        x_train = x_total[:split_index]
        x_test = x_total[split_index:]
        y_train = y_total[:split_index]
        y_test = y_total[split_index:]
        return x_train, x_test, y_train, y_test
    except Exception as err:
        # print(err, file=sys.stderr)
        return None







    