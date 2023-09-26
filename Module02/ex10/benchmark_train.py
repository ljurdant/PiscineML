import pandas as pd
import numpy as np

from utils.polynomial_model import add_polynomial_features
from utils.mylinearregressionmulti import MyLinearRegressionMulti
from utils.data_spliter import data_spliter
from datetime import datetime
import threading
from utils.utils import zscore

#Pasring
data = pd.read_csv("../data/space_avocado.csv")
target = zscore(np.array(data["target"]).reshape(-1, 1))
target_raw = np.array(data["target"]).reshape(-1, 1)

#Setting up polynomial features
weights = []
prod_distances = []
time_deliverys = []
weights_raw = []
prod_distances_raw = []
time_deliverys_raw = []

# for order in range(1, 5):
weights = add_polynomial_features(zscore(np.array(data["weight"]).reshape(-1, 1)), 4)
prod_distances = add_polynomial_features(zscore(np.array(data["prod_distance"]).reshape(-1, 1)), 4)
time_deliverys = add_polynomial_features(zscore(np.array(data["time_delivery"]).reshape(-1, 1)), 4)
weights_raw = add_polynomial_features(np.array(data["weight"]).reshape(-1, 1), 4)
prod_distances_raw = add_polynomial_features(np.array(data["prod_distance"]).reshape(-1, 1), 4)
time_deliverys_raw = add_polynomial_features(np.array(data["time_delivery"]).reshape(-1, 1), 4)


def run(y, weight, prod_distance, time_delivery, x_raw, y_raw, thetas, alpha, nb_iter):    


    x = np.concatenate((weight, prod_distance, time_delivery), axis = 1)
    

    x_train, _, y_train, _ = data_spliter(x, y, 0.5)
    _, x_test, _ , y_test = data_spliter(x_raw, y_raw, 0.5)

    #Training model
    lr = MyLinearRegressionMulti(thetas, alpha, nb_iter)
    try:
        lr.fit_(x_train, y_train)
    except Exception as error:
        print(error)
        return run(y, weight, prod_distance, time_delivery, thetas, alpha / 10, nb_iter)
    else:
        return lr.mse_(x_test, y_test)

#Open and create results file
now = datetime.now()
filename = "results/results_"+now.strftime("%H:%M:%S")
f = open(filename, "x")
lock = threading.Lock()
def thread_function(weight_order, prod_distance_order, time_delivery_order):
    lock.acquire()
    print("Starting ",weight_order, prod_distance_order, time_delivery_order)
    lock.release()
    thetas_weight = np.random.uniform(-1, 1, weight_order)
    thetas_order = np.random.uniform(-1, 1, prod_distance_order)
    thetas_delivery = np.random.uniform(-1, 1, time_delivery_order)
    thetas = np.concatenate((np.array([50000]), thetas_weight, thetas_delivery, thetas_order)).reshape(-1, 1)
    mse = 0
    # print(thetas)
    # x_raw =[]
    x_raw = np.concatenate((weights_raw[:,:weight_order], prod_distances_raw[:,:prod_distance_order], time_deliverys_raw[:,:time_delivery_order]), axis = 1)
    mse = run(target, weights[:,:weight_order], prod_distances[:,:prod_distance_order], time_deliverys[:,:time_delivery_order], x_raw, target_raw, thetas, 1e-7, 1000)
    
    lock.acquire()
    with open(filename, "a") as file:
        print(weight_order, prod_distance_order, time_delivery_order, "->", thetas, ":", mse, file=file)
    print(weight_order, prod_distance_order, time_delivery_order, "->", thetas, ":", mse)
    lock.release()


threads = []
for  weight_order in range(1, 5):
    for prod_distance_order in range(1, 5):
        for time_delivery_order in range(1, 5):
            x = threading.Thread(target=thread_function, args=(weight_order, prod_distance_order, time_delivery_order,))
            threads.append(x)
            x.start()
for thread in threads:
    thread.join()
f.close()