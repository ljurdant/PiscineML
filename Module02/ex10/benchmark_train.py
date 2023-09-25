import pandas as pd
import numpy as np

from polynomial_model import add_polynomial_features
from mylinearregressionmulti import MyLinearRegressionMulti
from data_spliter import data_spliter
from datetime import datetime
import threading
from utils import zscore

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

for order in range(1, 5):
    weights.append(add_polynomial_features(zscore(np.array(data["weight"]).reshape(-1, 1)), order))
    prod_distances.append(add_polynomial_features(zscore(np.array(data["prod_distance"]).reshape(-1, 1)), order))
    time_deliverys.append(add_polynomial_features(zscore(np.array(data["time_delivery"]).reshape(-1, 1)), order))
    weights_raw.append(add_polynomial_features(np.array(data["weight"]).reshape(-1, 1), order))
    prod_distances_raw.append(add_polynomial_features(np.array(data["prod_distance"]).reshape(-1, 1), order))
    time_deliverys_raw.append(add_polynomial_features(np.array(data["time_delivery"]).reshape(-1, 1), order))


def run(data, y, weight, prod_distance, time_delivery, x_raw, y_raw, thetas, alpha, nb_iter):    


    x = np.concatenate((weight, prod_distance, time_delivery), axis = 1)
    

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.5)

    #Training model
    lr = MyLinearRegressionMulti(thetas, alpha, nb_iter)
    try:
        lr.fit_(x_train, y_train)
    except Exception as error:
        print(error)
        return run(data, y, weight, prod_distance, time_delivery, thetas, alpha / 10, nb_iter)
    else:
        return lr.mse_(x_raw, y_raw)

#Open and create results file
now = datetime.now()
f = open("results/results_"+now.strftime("%H:%M:%S"), "x")    
lock = threading.Lock()
def thread_function(weight_order):
    thetas_weight = np.random.uniform(-1, 1, weight_order)
    for prod_distance_order in range(1, 5):
        thetas_order = np.random.uniform(-1, 1, prod_distance_order)
        for time_delivery_order in range(1, 5):
            print("Starting ",weight_order, prod_distance_order, time_delivery_order)
            thetas_delivery = np.random.uniform(-1, 1, time_delivery_order)
            thetas = np.concatenate((np.random.uniform(-1, 1, 1), thetas_weight, thetas_delivery, thetas_order)).reshape(-1, 1)
            mse = 0
            # print(thetas)
            # x_raw =[]
            x_raw = np.concatenate((weights_raw[weight_order - 1], prod_distances_raw[prod_distance_order - 1], time_deliverys_raw[time_delivery_order - 1]), axis = 1)
            mse = run(data, target, weights[weight_order - 1], prod_distances[prod_distance_order - 1], time_deliverys[time_delivery_order - 1], x_raw, target_raw, thetas, 1e-6, 1000)
            
            lock.acquire(f)
            print(weight_order, prod_distance_order, time_delivery_order, "->", thetas, ":", mse, file=f)
            lock.release(f)
            print(weight_order, prod_distance_order, time_delivery_order, "->", thetas, ":", mse,end=" ")
        

threads = []
for  order in range(1, 5):
    x = threading.Thread(target=thread_function, args=(order,))
    threads.append(x)
    x.start()
for thread in threads:
    thread.join()
f.close()