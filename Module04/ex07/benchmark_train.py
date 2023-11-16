import pandas as pd
import numpy as np
import sys

from utils.poylnomial_model_extended import add_polynomial_features
from utils.ridge import myRidge
from utils.data_spliter import data_spliter
from datetime import datetime
import threading
# from sklearn.linear_model import LinearRegression

from utils.zscore import zscore

# def xavier_init(shape):
#     # Assuming shape is the shape of the weight matrix for a layer
#     fan_in, fan_out = shape[0], shape[1]
#     scale = np.sqrt(2.0 / (fan_in + fan_out))
#     return np.random.randn(*shape) * scale



def thread_function(weight_order, prod_distance_order, time_delivery_order, lambda_:float, thetas = np.array([])):

    loss = 0
    lock.acquire()
    print("Starting ",weight_order, prod_distance_order, time_delivery_order, lambda_)
    lock.release()
    
    x_train_ = np.concatenate((x_train[:,:weight_order],x_train[:,4:4+prod_distance_order],x_train[:,8:8+time_delivery_order]), axis=1)
    x_test_ = np.concatenate((x_test[:,:weight_order],x_test[:,4:4+prod_distance_order],x_test[:,8:8+time_delivery_order]), axis=1)
    if thetas.shape == (0,):
        regularization_matrix = lambda_* np.identity(x_train_.shape[1])
        X_transpose_X = np.dot(x_train_.T, x_train_)
        coefficients = np.linalg.inv(X_transpose_X + regularization_matrix).dot(x_train_.T).dot(y_train)
        thetas = np.concatenate(([[1]],coefficients.reshape(-1,1)), axis=0)
        print(thetas)

    try:
        loss, new_thetas = run(x_train_,y_train,x_test_,y_test,thetas,lambda_)
        lock.acquire()
        # print(new_thetas)
        models.append([loss, weight_order, prod_distance_order, time_delivery_order,lambda_, new_thetas])
        print(weight_order, prod_distance_order, time_delivery_order, lambda_, ":", loss)
        lock.release()
    except Exception as error:
        print(error, sys.stderr)
   

def run(x_train,y_train,x_test,y_test,thetas, lambda_:float, alpha = 1e-6, nb_iter = 100000):    
    #Training model
    lr = myRidge(thetas, alpha, nb_iter, lambda_)
    new_thetas = lr.fit_(x_train, y_train.reshape(-1,1))
    return round(lr.loss_(y_test, lr.predict_(x_test))), new_thetas

#Parsing
data = pd.read_csv("../data/space_avocado.csv")
target = np.array(data["target"]).reshape(-1,1)

#Setting up polynomial features
weights = []
prod_distances = []
time_deliverys = []

data_raw = np.array(data, copy=True)
data = np.array(data)

weights = add_polynomial_features(((data[:,0])).reshape(-1,1), 4)
prod_distances = add_polynomial_features(((data[:,1])).reshape(-1,1), 4)
time_deliverys = add_polynomial_features(((data[:,2])).reshape(-1,1), 4)

weights_raw = add_polynomial_features(((data_raw[:,0])).reshape(-1,1), 4)
prod_distances_raw = add_polynomial_features(((data_raw[:,1])).reshape(-1,1), 4)
time_deliverys_raw = add_polynomial_features(((data_raw[:,2])).reshape(-1,1), 4)


x = np.concatenate((weights, prod_distances, time_deliverys),axis=1)
x_raw = np.concatenate((weights_raw, prod_distances_raw, time_deliverys_raw),axis=1)
x_train, _, y_train, _ = data_spliter(x, data[:,3].reshape(-1,1), 0.6)
_, x_test, _, y_test = data_spliter(x_raw, data_raw[:,3].reshape(-1,1), 0.6)

# x_train = zscore(x_train)
# y_train = zscore(y_train)

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
                for lambda_ in [0 ]:
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
