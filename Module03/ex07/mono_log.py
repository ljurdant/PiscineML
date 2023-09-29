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


def parse_zipcode():
    if len(sys.argv) <= 1:
        raise Exception("missing argument")

    valid_zips = ["0", "1", "2", "3"]

    zip_arg = sys.argv[1]

    arg_zip = zip_arg.split("=")
    if (arg_zip[0] != "-zipcode"):
        raise Exception("unknown flag: "+arg_zip[0])
    if (len(arg_zip) < 2):
        raise Exception("no value provide for zipcode")
    if (len(arg_zip) > 2) or (arg_zip[1] not in valid_zips):
        raise Exception("bad value for zipcode")

    return int(arg_zip[1])

zipcode = -1
try:
    zipcode = parse_zipcode()
except Exception as error:
    print("Error: "+str(error))
    print("Usage: python mono_log.py -zipcode=x")

if (zipcode != -1):
    solar_data = pd.read_csv("../data/solar_system_census.csv")
    planet_data = pd.read_csv("../data/solar_system_census_planets.csv")

    x = np.array(solar_data[["height","weight","bone_density"]])
    y = planet_data["Origin"] == zipcode
    y = np.array(y.astype({"Origin": "int"})).reshape(-1, 1)

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.5)
    # print(zscore(x_train[:,0]))
    # x_train = np.concatenate((minmax(x_train[:,0]).reshape(-1,1), minmax(x_train[:,1]).reshape(-1,1), (x_train[:,2]).reshape(-1,1)), axis=1)
    # x_train = minmax(x_train)
    # print(x_train)
    thetas = np.array([[10.0],[1],[1],[1]])

    myLr = MyLogisticRegression(thetas, max_iter=100000)
    myLr.fit_(x_train, y_train.reshape(-1, 1))
    print("Thetas = ",myLr.theta)
    # print(x_train)
    print("Loss = ",myLr.loss_(x_test, y_test.reshape(-1, 1)))

    y_hat = myLr.predict_(x_test)
    count = 0
    # print(y_hat)
    y_hat = y_hat >= 0.5
    for y_pred, y_true in zip(y_hat, y_test):
        count+=(y_pred[0] == y_true)
    accuracy = count / y_test.shape[0]

    pred_data = x_test[y_hat[:,0]]
    other_data = x_test[y_hat[:,0] == False]
    print("Accuracy = "+str(round(accuracy, 4)*100)+"%")
    fig, axs = plt.subplots(2,2)
    axs[0,0].scatter(other_data[:,0],other_data[:,1])
    axs[0,0].scatter(pred_data[:,0],pred_data[:,1])
    axs[0,0].set_ylabel("height")
    axs[1,0].scatter(other_data[:,0], other_data[:,2])
    axs[1,0].scatter(pred_data[:,0],pred_data[:,2])
    axs[1,0].set_ylabel("bone_density")
    axs[1,0].set_xlabel("weight")
    axs[1,1].scatter(other_data[:,1], other_data[:,2])
    axs[1,1].scatter(pred_data[:,1], pred_data[:,2])
    axs[1,1].set_xlabel("height")
    
    plt.show()
