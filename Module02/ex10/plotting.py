import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils.polynomial_model import add_polynomial_features
from utils.mylinearregressionmulti import MyLinearRegressionMulti


data = pd.read_csv("../data/space_avocado.csv")

target = np.array(data["target"]).reshape(-1, 1)
weight = np.array(data["weight"]).reshape(-1, 1)
prod_distance = np.array(data["prod_distance"]).reshape(-1, 1)
time_delivery = np.array(data["time_delivery"]).reshape(-1, 1)

# def plot_comparison(thetas, weight_order, prod_distance_order, time_delivery_order):
#     weights = add_polynomial_features(weight, weight_order)
#     prod_distances = add_polynomial_features(prod_distance, prod_distance_order)
#     time_deliverys = add_polynomial_features(time_delivery, time_delivery_order)
    
#     x = np.concatenate((weights, prod_distances, time_deliverys), axis = 1)



#     lr = MyLinearRegressionMulti(thetas)
#     y_hat = lr.predict_(x)

fig, axs = plt.subplots(2, 2,sharey="row")
fig.subplots_adjust(hspace=.5)
axs[0,0].scatter(weight, target)
axs[0,0].set_xlabel("weight")
axs[0,1].scatter(prod_distance, target)
axs[0,1].set_xlabel("prod_distance")
axs[1,0].scatter(time_delivery, target)
axs[1,0].set_xlabel("time_delivery")
plt.show()