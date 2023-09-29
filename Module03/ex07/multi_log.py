import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression

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
    if np.mean(x) and np.std(x):
        return (x - np.mean(x)) / np.std(x)

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
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def data_spliter(x, y, proportion):
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

def classifier(zipcode, x_train, y_train):
    y = y_train == zipcode
    y = np.array(y.astype({"Origin": "int"})).reshape(-1, 1)

    thetas = np.array([[10.0],[1],[1],[1]])

    max_iter = 100000
    if zipcode == 2:
        max_iter*=10
    myLr = MyLogisticRegression(thetas, max_iter=max_iter)
    myLr.fit_(x_train, y_train.reshape(-1, 1))

solar_data = pd.read_csv("../data/solar_system_census.csv")
planet_data = pd.read_csv("../data/solar_system_census_planets.csv")

y = planet_data["Origin"]
x = np.array(solar_data[["height","weight","bone_density"]])

x_train, x_test, y_train, y_test = data_spliter(x, y, 0.5)


