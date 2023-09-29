import pandas as pd
import numpy as np
import sys

from utils.polynomial_model import add_polynomial_features
from utils.mylinearregressionmulti import MyLinearRegressionMulti
from utils.data_spliter import data_spliter
from datetime import datetime
import threading
from utils.utils import zscore, zscore_multi, minmax

#Parsing
data_raw = pd.read_csv("../data/space_avocado.csv")
data = (np.array(data_raw[["weight","prod_distance","time_delivery","target"]]))
target = np.array(data[:,3]).reshape(-1, 1)
target_raw = np.array(data_raw["target"]).reshape(-1, 1)

#Setting up polynomial features
weights = []
prod_distances = []
time_deliverys = []
weights_raw = []
prod_distances_raw = []
time_deliverys_raw = []

# for order in range(1, 5):
weights = add_polynomial_features((np.array(data[:,0])).reshape(-1, 1), 4)
prod_distances = add_polynomial_features((np.array(data[:,1])).reshape(-1, 1), 4)
time_deliverys = add_polynomial_features((np.array(data[:,2])).reshape(-1, 1), 4)
weights_raw = add_polynomial_features(np.array(data_raw["weight"]).reshape(-1, 1), 4)
prod_distances_raw = add_polynomial_features(np.array(data_raw["prod_distance"]).reshape(-1, 1), 4)
time_deliverys_raw = add_polynomial_features(np.array(data_raw["time_delivery"]).reshape(-1, 1), 4)

def run(y, weight, prod_distance, time_delivery, x_raw, y_raw, thetas, alpha = 0.001, nb_iter = 1000):    

    x = np.concatenate((weight, prod_distance, time_delivery), axis = 1)
    x_train, _, y_train, _ = data_spliter(x, y, 0.5)
    _, x_test, _ , y_test = data_spliter(x_raw, y_raw, 0.5)

    #Training model
    lr = MyLinearRegressionMulti(thetas, alpha, nb_iter)
    new_thetas = lr.fit_(x_train, y_train.reshape(-1,1))
    return round(lr.mse_(x_test, y_test.reshape(-1,1))), new_thetas
        # return run(y, weight, prod_distance, time_delivery, thetas, alpha/10, nb_iter)

#Open and create results file
now = datetime.now()
filename = "models/model_"+now.strftime("%m_%d_%H:%M:%S")+".csv"
f = open(filename, "x")
print("mse,weight_order,prod_distance_order,time_delivery_order,thetas", file=f)
f.close()
lock = threading.Lock()

def thread_function(weight_order, prod_distance_order, time_delivery_order, thetas = np.array([])):
    if thetas.shape == (0,):
        thetas_weight = np.random.uniform(0, 1, weight_order)
        thetas_order = np.random.uniform(0, 1, prod_distance_order)
        thetas_delivery = np.random.uniform(0, 1, time_delivery_order)
        thetas = np.concatenate(( np.random.uniform(0, 1, 1), thetas_weight, thetas_delivery, thetas_order)).reshape(-1, 1)
    mse = 0
    lock.acquire()
    print("Starting ",weight_order, prod_distance_order, time_delivery_order, thetas)
    lock.release()
    # x_raw =[]
    x_raw = np.concatenate((weights_raw[:,:weight_order], prod_distances_raw[:,:prod_distance_order], time_deliverys_raw[:,:time_delivery_order]), axis = 1)
    
    try:
        mse, new_thetas = run(target, weights[:,:weight_order], prod_distances[:,:prod_distance_order], time_deliverys[:,:time_delivery_order], x_raw, target_raw, thetas, 1e-7, 100000)
    except Exception as error:
        print(error, sys.stderr)

    lock.acquire()
    with open(filename, "a") as file:
        print(mse, weight_order, prod_distance_order, time_delivery_order, " ".join([str(theta[0]) for theta in new_thetas]),sep=",",file=file)

    print(weight_order, prod_distance_order, time_delivery_order, "->", new_thetas, ":", mse)
    lock.release()


threads = []
if len(sys.argv) >= 2:
    model_filename = sys.argv[1]
    try:
        model_data = pd.read_csv(model_filename)
        for row_index in range(model_data.shape[0]):
            thetas = np.array([float(theta)for theta in model_data["thetas"][row_index].split()]).reshape(-1, 1)
            # print(model_data["weight_order"][row_index], 
                # model_data["prod_distance_order"][row_index], 
                # model_data["time_delivery_order"][row_index],
                # thetas)
            x = threading.Thread(target=thread_function, args=(
                model_data["weight_order"][row_index], 
                model_data["prod_distance_order"][row_index], 
                model_data["time_delivery_order"][row_index],
                thetas))
            threads.append(x)
            x.start()
            

    except Exception as error:
        print(error, file=sys.stderr)
    

else:
    for  weight_order in range(1, 5):
        for prod_distance_order in range(1, 5):
            for time_delivery_order in range(1, 5):
                x = threading.Thread(target=thread_function, args=(weight_order, prod_distance_order, time_delivery_order,))
                threads.append(x)
                x.start()
for thread in threads:
    thread.join()

