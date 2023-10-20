import pandas as pd
import numpy as np
import sys

from utils.poylnomial_model_extended import add_polynomial_features
from utils.ridge import myRidge
from utils.data_spliter import data_spliter
from datetime import datetime
import threading

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

def run(
        y, 
        weight, 
        prod_distance, 
        time_delivery, 
        x_raw, 
        y_raw, 
        thetas, 
        lambda_:float, 
        alpha = 1e-6, 
        nb_iter = 1000000000):    

    x = np.concatenate((weight, prod_distance, time_delivery), axis = 1)
    x_train, _, y_train, _ = data_spliter(x, y, 0.7)
    _, x_test, _ , y_test = data_spliter(x_raw, y_raw, 0.7)

    #Training model
    lr = myRidge(thetas, alpha, nb_iter, lambda_)
    new_thetas = lr.fit_(x_train, y_train.reshape(-1,1))
    return round(lr.loss_(x_test, y_test.reshape(-1,1))), new_thetas, lr.diverge

#Open and create results file
now = datetime.now()
filename = "models/model_"+now.strftime("%m_%d_%H:%M:%S")+".csv"
f = open(filename, "x")
print("loss,weight_order,prod_distance_order,time_delivery_order,lambda,thetas,diverge", file=f)
f.close()
lock = threading.Lock()

def thread_function(weight_order, prod_distance_order, time_delivery_order, lambda_:float, thetas = np.array([])):
    if thetas.shape == (0,):
        thetas_weight = np.random.uniform(0, 1, weight_order)
        thetas_order = np.random.uniform(0, 1, prod_distance_order)
        thetas_delivery = np.random.uniform(0, 1, time_delivery_order)
        thetas = np.concatenate(([1],
                                 thetas_weight, 
                                 thetas_delivery, 
                                 thetas_order)
                                ).reshape(-1, 1)
    loss = 0
    lock.acquire()
    print("Starting ",weight_order, prod_distance_order, time_delivery_order, lambda_, thetas)
    lock.release()
    # x_raw =[]
    x_raw = np.concatenate(
        (weights_raw[:,:weight_order], 
         prod_distances_raw[:,:prod_distance_order], 
         time_deliverys_raw[:,:time_delivery_order]
        ),
        axis = 1)
    
    try:
        loss, new_thetas, diverge = run(
            target, 
            weights[:,:weight_order], 
            prod_distances[:,:prod_distance_order], 
            time_deliverys[:,:time_delivery_order], 
            x_raw, 
            target_raw, 
            thetas,
            lambda_
            )
    except Exception as error:
        print(error, sys.stderr)
   
    lock.acquire()
    with open(filename, "a") as file:
        print(loss, weight_order, prod_distance_order, time_delivery_order,lambda_," ".join([str(theta[0]) for theta in new_thetas]),diverge,sep=",",file=file)

    print(weight_order, prod_distance_order, time_delivery_order, lambda_, diverge,"->", new_thetas, ":", loss)
    lock.release()


threads = []
if len(sys.argv) >= 2:
    model_filename = sys.argv[1]
    try:
        model_data = pd.read_csv(model_filename)
        if len(sys.argv) >= 3:
            nb_models = int(sys.argv[2])
            model_data = model_data.sort_values(by=["loss"], key = lambda s: s.str[:].astype(float))
            model_data = model_data.iloc[:nb_models]
        for index, row in model_data.iterrows():
            thetas = np.array([float(theta)for theta in row["thetas"].split()]).reshape(-1, 1)
            
            if row["diverge"] == False:
                x = threading.Thread(target=thread_function, args=(
                    row["weight_order"], 
                    row["prod_distance_order"], 
                    row["time_delivery_order"],
                    row["lambda"],
                    thetas))

                threads.append(x)
                x.start()
            

    except Exception as error:
        print(error, file=sys.stderr)
    

else:
    for  weight_order in range(1, 5):
        for prod_distance_order in range(1, 5):
            for time_delivery_order in range(1, 5):
                for lambda_ in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    thread = threading.Thread(target=thread_function, args=(weight_order, prod_distance_order, time_delivery_order,lambda_))
                    threads.append(thread)
                    thread.start()
for thread in threads:
    thread.join()

