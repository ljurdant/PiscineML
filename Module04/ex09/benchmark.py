import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import threading

from my_logistic_regression import MyLogisticRegression
from poylnomial_model_extended import add_polynomial_features
from other_metrics import f1_score_

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
        y_train = y_total[:split_index].reshape(-1,1)
        y_test = y_total[split_index:].reshape(-1,1)
        return x_train, x_test, y_train, y_test
    except Exception as err:
        print(err, file=sys.stderr)
        return None

def classifier(zipcode, x_train, y_train, lambda_, theta_length):

    thetas = np.ones((theta_length+1,1))
    y_train = y_train == zipcode


    max_iter = 10000
    myLr = MyLogisticRegression(thetas, max_iter=max_iter, lambda_=lambda_, alpha=1e-5)
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
models = []
def thread_function(height_order, weight_order, bone_density_order, lambda_:float, models):

    
    lock.acquire()
    print("Starting ",height_order, weight_order, bone_density_order, lambda_)
    lock.release()
    
    x_train_ = np.concatenate((x_train[:,:height_order],x_train[:,3:3+weight_order],x_train[:,6:6+bone_density_order]), axis=1)
    x_test_ = np.concatenate((x_test[:,:height_order],x_test[:,3:3+weight_order],x_test[:,6:6+bone_density_order]), axis=1)

    theta_length = height_order+weight_order+bone_density_order

    thetas0 = classifier(0, x_train_, y_train, lambda_, theta_length)
    thetas1 = classifier(1, x_train_, y_train, lambda_, theta_length)
    thetas2 = classifier(2, x_train_, y_train, lambda_, theta_length)
    thetas3 = classifier(3, x_train_, y_train, lambda_, theta_length)


    y_hat = predicter(thetas0, thetas1, thetas2, thetas3, x_test_)

    f1score = f1_score_(y, y_hat)
    lock.acquire()
    models.append([f1score, height_order, weight_order, bone_density_order, lambda_])
    lock.release()



solar_data = pd.read_csv("../data/solar_system_census.csv")
planet_data = pd.read_csv("../data/solar_system_census_planets.csv")


y = planet_data["Origin"]
y = np.array(y.astype({"Origin": "int"})).reshape(-1, 1)
# x = np.array(solar_data[["height","weight","bone_density"]])

heights = add_polynomial_features((np.array(solar_data[["height"]])).reshape(-1,1), 3)
weights = add_polynomial_features((np.array(solar_data[["weight"]])).reshape(-1,1), 3)
bone_densities = add_polynomial_features((np.array(solar_data[["bone_density"]])).reshape(-1,1), 3)

x = np.concatenate((heights, weights, bone_densities),axis=1)
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)

now = datetime.now()
lock = threading.Lock()

threads = []

if len(sys.argv) >= 2:
    model_filename = sys.argv[1]
    try:
        model_data = pd.read_csv(model_filename)
        if len(sys.argv) >= 3:
            nb_models = int(sys.argv[2])
            model_data = model_data.sort_values(by=["f1score"], key = lambda s: s.str[:].astype(float))
            model_data = model_data.iloc[:nb_models]
        for index, row in model_data.iterrows():
            thetas = np.array([float(theta)for theta in row["thetas"].split()]).reshape(-1, 1)
            thread = threading.Thread(target=thread_function, args=(
                row.at(["height_order"]), 
                row.at(["weight_order"]), 
                row.at(["bone_density_order"]),
                row.at(["lambda"]),
                models))

            threads.append(thread)
            thread.start()
    except Exception as error:
        print(error, file=sys.stderr)
else:
    for height_order in range(1, 4):
        for weight_order in range(1, 4):
            for bone_density_order in range(1, 4):
                for lambda_ in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    thread = threading.Thread(target=thread_function, args=(height_order, weight_order, bone_density_order,lambda_, models))
                    threads.append(thread)
                    thread.start()

for thread in threads:
    thread.join()

filename = "models/model_"+now.strftime("%m_%d_%H:%M:%S")+".csv"
f = open(filename, "x")
print("f1score,height_order,weight_order,bone_density_order,lambda,thetas", file=f)
f.close()

models.sort(key= lambda x: (x[1],x[2],x[3],x[4]))
for model in models:
    (loss, height_order, weight_order, bone_density_order, lambda_) = model
    with open(filename, "a") as file:
        print(loss, height_order, weight_order, bone_density_order,lambda_,file=file)

