import pandas as pd
import numpy as np

from polynomial_model import add_polynomial_features
from mylinearregressionmulti import MyLinearRegressionMulti
from data_spliter import data_spliter

data = pd.read_csv("../data/space_avocado.csv")

target = np.array(data["target"]).reshape(-1, 1)

def run(data, y, order_weight, order_prod_distance, order_time_delivery, thetas):
    #Data formatting
    weight = add_polynomial_features(np.array(data["weight"]).reshape(-1, 1), order_weight)
    prod_distance = add_polynomial_features(np.array(data["prod_distance"]).reshape(-1, 1), order_prod_distance)
    time_delivery = add_polynomial_features(np.array(data["time_delivery"]).reshape(-1, 1), order_time_delivery)

    x = np.append(weight, prod_distance, axis = 1)
    x = np.append(x, time_delivery, axis = 1)

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.5)

    #Training model
    lr = MyLinearRegressionMulti(thetas, 1e-7, 1000)
    lr.fit_(x_train, y_train)
    return lr.mse_(x_test, y_test)


# for weight_order in range(1, 5):
    # for prod_distance_order in range(1, 5):
        # for time_delivery_order in range(1, 5):
            # print(weight_order, prod_distance_order, time_delivery_order, ":", end=" ")
# mse = run(data, target, weight_order, prod_distance_order, time_delivery_order)
mse = run(data, target, 1, 1, 1, [[10000],[500], [10], [100]])
print("1 1 1 :",mse)
mse = run(data, target, 2, 1, 1, [[10000], [-1], [500], [10], [100]])
print("2 1 1 :",mse)

        