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
        print(err, file=sys.stderr)
        return None

def classifier(zipcode, x_train, y_train):

    thetas = np.array([[10.0],[1],[1],[1]])
    y_train = y_train == zipcode

    max_iter = 1000000
    myLr = MyLogisticRegression(thetas, max_iter=max_iter)
    return myLr.fit_(x_train, y_train.reshape(-1, 1))

def predicter(thetas1, thetas2, thetas3, thetas4, x_test):
    myLr1 = MyLogisticRegression(thetas1)
    y1 = myLr1.predict_(x_test)

    myLr2 = MyLogisticRegression(thetas2)
    y2 = myLr2.predict_(x_test)
    
    myLr3 = MyLogisticRegression(thetas3)
    y3 = myLr3.predict_(x_test)

    myLr4 = MyLogisticRegression(thetas4)
    y4 = myLr4.predict_(x_test)

    y_all = np.concatenate((y1,y2,y3,y4), axis=1)
    return np.argmax(y_all, axis=1)

solar_data = pd.read_csv("../data/solar_system_census.csv")
planet_data = pd.read_csv("../data/solar_system_census_planets.csv")

y = planet_data["Origin"]
y = np.array(y.astype({"Origin": "int"})).reshape(-1, 1)
x = np.array(solar_data[["height","weight","bone_density"]])

x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)

thetas0 = classifier(0, x_train, y_train)
thetas1 = classifier(1, x_train, y_train)
thetas2 = classifier(2, x_train, y_train)
thetas3 = classifier(3, x_train, y_train)

y_hat = predicter(thetas0, thetas1, thetas2, thetas3, x_test)
correct_count = np.count_nonzero(y_hat == y_test)
accuracy = correct_count / y_test.shape[0]
print("Accuracy = " + str(round(accuracy, 4)*100)+"%")
x0 = x_test[y_hat == 0]
x1 = x_test[y_hat == 1]
x2 = x_test[y_hat == 2]
x3 = x_test[y_hat == 3]

fig, axs = plt.subplots(2,2)
for x in [x0, x1, x2, x3]:
    axs[0,0].scatter(x[:,0],x[:,1])
    axs[0,0].set_ylabel("height")
    axs[1,0].scatter(x[:,0], x[:,2])
    axs[1,0].set_ylabel("bone_density")
    axs[1,0].set_xlabel("weight")
    axs[1,1].scatter(x[:,1], x[:,2])
    axs[1,1].set_xlabel("height")
    plt.legend(["zip=0","zip=1","zip=2","zip=3"])
plt.show()

