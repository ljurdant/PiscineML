import pandas as pd
import numpy as np
import sys

from utils.poylnomial_model_extended import add_polynomial_features
from utils.ridge import myRidge
from utils.data_spliter import data_spliter
from datetime import datetime
import threading

from utils.zscore import zscore




def thread_function(weight_order, prod_distance_order, time_delivery_order, lambda_:float, thetas = np.array([])):

    loss = 0
    lock.acquire()
    print("Starting ",weight_order, prod_distance_order, time_delivery_order, lambda_)
    lock.release()
    
    x_train_ = np.concatenate((x_train[:,:weight_order],x_train[:,4:4+prod_distance_order],x_train[:,8:8+time_delivery_order]), axis=1)
    x_test_ = np.concatenate((x_test[:,:weight_order],x_test[:,4:4+prod_distance_order],x_test[:,8:8+time_delivery_order]), axis=1)
    if thetas.shape == (0,):
        thetas_weight = np.random.uniform(0, 1, weight_order)
        thetas_prod_distance = np.random.uniform(0, 1, prod_distance_order)
        thetas_time_delivery = np.random.uniform(0, 1, time_delivery_order)
        thetas = np.concatenate(([1],thetas_weight, thetas_prod_distance, thetas_time_delivery )).reshape(-1, 1)

    try:
        loss, new_thetas = run(x_train_,y_train,x_test_,y_test,thetas,lambda_)
        lock.acquire()
        # print(new_thetas)
        models.append([loss, weight_order, prod_distance_order, time_delivery_order,lambda_, new_thetas])
        print(weight_order, prod_distance_order, time_delivery_order, lambda_, ":", loss)
        lock.release()
    except Exception as error:
        print(error, sys.stderr)
   

def run(x_train_,y_train,x_test_,y_test,thetas, lambda_:float, alpha = 1e-3, nb_iter = 2000):    
    #Training model
    lr = myRidge(thetas, alpha, nb_iter, lambda_)
    new_thetas = lr.fit_(x_train_, y_train.reshape(-1,1))
    
    return round(lr.loss_(y_test, lr.predict_(x_test_))), new_thetas

#Parsing
data = pd.read_csv("../data/space_avocado.csv")
target = np.array(data["target"]).reshape(-1,1)

#Setting up polynomial features
weights = []
prod_distances = []
time_deliverys = []

data = np.array(data, copy=True)

weights = zscore(add_polynomial_features((data[:,0]).reshape(-1,1), 4))
prod_distances = zscore(add_polynomial_features((data[:,1]).reshape(-1,1), 4))
time_deliverys = zscore(add_polynomial_features((data[:,2]).reshape(-1,1), 4))

x = np.concatenate((weights, prod_distances, time_deliverys),axis=1)
x_train, x_test, y_train, y_test = data_spliter(x, data[:,3].reshape(-1,1), 0.6)


now = datetime.now()
lock = threading.Lock()
models = []
threads = []

if len(sys.argv) >= 2:
    model_filename = sys.argv[1]
    try:
        model_data = pd.read_csv(model_filename)
        if len(sys.argv) >= 3:
            nb_models = int(sys.argv[2])
            model_data = model_data.sort_values(by=["loss"], key = lambda s: s.astype(float))
            model_data = model_data.iloc[:nb_models]
        for index, row in model_data.iterrows():
            thetas = np.array([float(theta)for theta in row["thetas"].split()]).reshape(-1, 1)
            thread = threading.Thread(target=thread_function, args=(
                row["weight_order"], 
                row["prod_distance_order"], 
                row["time_delivery_order"],
                row["lambda"],
                thetas))

            threads.append(thread)
            thread.start()
            

    except Exception as error:
        print(error, file=sys.stderr)
else:
    for  weight_order in range(1, 2):
        for prod_distance_order in range(1, 2):
            for time_delivery_order in range(1, 2):
                for lambda_ in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0 ]:
                    thread = threading.Thread(target=thread_function, args=(weight_order, prod_distance_order, time_delivery_order,lambda_))
                    threads.append(thread)
                    thread.start()
for thread in threads:
    thread.join()


filename = "models/model_"+now.strftime("%m_%d_%H:%M:%S")+".csv"
f = open(filename, "x")
print("loss,weight_order,prod_distance_order,time_delivery_order,lambda,thetas", file=f)
f.close()
models.sort(key= lambda x: (x[1],x[2],x[3],x[4]))
for model in models:
    [loss, weight_order, prod_distance_order, time_delivery_order, lambda_, new_thetas] = model
    with open(filename, "a") as file:
        print(loss, weight_order, prod_distance_order, time_delivery_order,lambda_," ".join([str(theta[0]) for theta in new_thetas]),sep=",",file=file)