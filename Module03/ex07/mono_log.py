import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression

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

    valid_zips = ["1", "2", "3", "4"]

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

    # thetas = np.append([10.0],np.random.uniform(-1,1,3)).reshape(-1, 1)
    thetas = np.array([[11.03593032],[-0.06548079],[ -0.01847231],[ 3.08083253]])
    myLr = MyLogisticRegression(thetas, max_iter=10000)
    myLr.fit_(x_train, y_train.reshape(-1, 1))
    print("Thetas = ",myLr.theta)
    print("Loss = ",myLr.loss_(x_test, y_test.reshape(-1, 1)))
    y_hat = myLr.predict_(x_test)
    # print(y_test.astype(int) == y_hat.astype(int))
    count = 0
    for y_pred, y_true in zip(y_hat, y_test):
        y_pred = round(y_pred[0])
        count+=(y_pred == y_true)
    accuracy = count / y_test.shape[0]
    print("Accuracy = "+str(round(accuracy, 4)*100)+"%")
