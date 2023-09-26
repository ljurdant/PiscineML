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
filename = "results/results_2_"+now.strftime("%H:%M:%S")
f = open(filename, "x")
lock = threading.Lock()
def thread_function(weight_order, prod_distance_order, time_delivery_order, thetas):
    lock.acquire()
    print("Starting ",weight_order, prod_distance_order, time_delivery_order)
    lock.release()
   
    # print(thetas)
    # x_raw =[]
    x_raw = np.concatenate((weights_raw[:,:weight_order], prod_distances_raw[:,:prod_distance_order], time_deliverys_raw[:,:time_delivery_order]), axis = 1)
    mse = run(target, weights[:,:weight_order], prod_distances[:,:prod_distance_order], time_deliverys[:,:time_delivery_order], x_raw, target_raw, thetas, 1e-7, 2000)
    
    lock.acquire()
    with open(filename, "a") as file:
        print(weight_order, prod_distance_order, time_delivery_order, "->", thetas, ":", mse, file=file)
    print(weight_order, prod_distance_order, time_delivery_order, "->", thetas, ":", mse)
    lock.release()


threads = []

# x = threading.Thread(target=thread_function, args=(1, 1, 4,[[ 5.00000000e+04], [-3.80965663e-01]
# , [ 5.41726486e-01]
# , [ 2.33259930e-01]
# , [ 9.15785506e-01]
# , [-4.97755303e-01]
# , [ 6.95446840e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(1, 1, 3,[[ 5.00000000e+04],
#  [ 3.73820783e-01],
#  [-9.91496268e-01],
#  [-7.63294286e-01],
#  [-7.71337363e-01],
#  [ 1.62637357e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(1, 1, 1,[[ 5.00000000e+04],
#  [ 7.33018073e-01],
#  [-3.98684627e-01],
#  [-4.04230218e-02]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(1, 1, 2,[[ 5.00000000e+04],
#  [ 3.37894124e-01],
#  [ 8.18597645e-01],
#  [-2.35653711e-02],
#  [ 8.97003435e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(2, 1, 1,[[ 5.00000000e+04],
#  [-3.90167329e-01],
#  [-3.30531860e-01],
#  [-1.56352821e-01],
#  [-1.24651530e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(3, 1, 2,[[ 5.00000000e+04],
#  [-5.86062860e-01],
#  [ 2.93106742e-01],
#  [ 7.99641670e-01],
#  [-4.30065488e-01],
#  [-7.57869513e-01],
#  [-2.01578521e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(2, 1, 3,[[ 5.00000000e+04],
#  [ 6.16259635e-01],
#  [-7.17282395e-01],
#  [-3.26988880e-01],
#  [-3.60991146e-01],
#  [-9.42286778e-02],
#  [ 4.03410133e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(2, 1, 4,[[ 5.00000000e+04],
#  [-5.86626124e-01],
#  [ 6.84136051e-01],
#  [-9.97704383e-01],
#  [ 2.57302643e-01],
#  [-1.32630581e-01],
#  [ 4.25415245e-01],
#  [-6.43635866e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(3, 1, 1,[[ 5.00000000e+04],
#  [ 6.33838152e-01],
#  [ 3.32943840e-01],
#  [-4.43191100e-01],
#  [ 3.67675792e-01],
#  [ 8.80370504e-01]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(3, 1, 3,[[ 5.00000000e+04],
#  [-1.37282592e-01],
#  [-2.85248557e-01],
#  [-3.95114614e-01],
#  [-4.89495138e-02],
#  [ 1.09401266e-01],
#  [-4.25983482e-01],
#  [ 7.72326137e-02]]))
# threads.append(x)
# x.start()

# x = threading.Thread(target=thread_function, args=(2, 1, 2,[[ 5.00000000e+04],
#  [-6.59652712e-02],
#  [-9.91902070e-01],
#  [-9.06307967e-01],
#  [-6.79833022e-01],
#  [-8.54291591e-01]]))
# threads.append(x)
# x.start()


x = threading.Thread(target=thread_function, args=(2, 1, 1,[[50000.0], [-0.390167329], [-0.33053186], [-0.156352821], [-0.12465153]]))
threads.append(x)
x.start()

x = threading.Thread(target=thread_function, args=(1, 1, 2,[[50000.0], [0.337894124], [0.818597645], [-0.0235653711], [0.897003435]]))
threads.append(x)
x.start()

x = threading.Thread(target=thread_function, args=(3, 1, 2,[[50000.0], [-0.58606286], [0.293106742], [0.79964167], [-0.430065488], [-0.757869513], [-0.201578521]]))
threads.append(x)
x.start()

x = threading.Thread(target=thread_function, args=(1, 1, 3,[[50000.0], [0.373820783], [-0.991496268], [-0.763294286], [-0.771337363], [0.162637357]]))
threads.append(x)
x.start()

x = threading.Thread(target=thread_function, args=(1, 1, 1,[[50000.0], [0.733018073], [-0.398684627], [-0.0404230218]]))
threads.append(x)
x.start()


for thread in threads:
    thread.join()
f.close()