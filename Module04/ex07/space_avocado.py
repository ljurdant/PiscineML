import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.poylnomial_model_extended import add_polynomial_features
from utils.ridge import myRidge

data = pd.read_csv("../data/space_avocado.csv")

target = np.array(data["target"]).reshape(-1, 1)
weight = np.array(data["weight"]).reshape(-1, 1)
prod_distance = np.array(data["prod_distance"]).reshape(-1, 1)
time_delivery = np.array(data["time_delivery"]).reshape(-1, 1)

def plot_comparison(weight_order, prod_distance_order, time_delivery_order, thetas, lambda_):
    weights = add_polynomial_features(weight, weight_order)
    prod_distances = add_polynomial_features(prod_distance, prod_distance_order)
    time_deliverys = add_polynomial_features(time_delivery, time_delivery_order)
    
    x = np.concatenate((weights, prod_distances, time_deliverys), axis = 1)



    lr = myRidge(thetas, lambda_=lambda_)
    y_hat = lr.predict_(x)
    fig, axs = plt.subplots(3, 2,sharey="row")
    fig.subplots_adjust(hspace=.5)
    fig.subplots_adjust(wspace=0)
    axs[0,0].scatter(weight, target)
    axs[0,1].scatter(weight, y_hat, color="orange")
    axs[0,0].set_xlabel("weight")
    axs[0,1].set_xlabel("weight")
    axs[1,0].scatter(prod_distance, target)
    axs[1,1].scatter(prod_distance, y_hat, color="orange")
    axs[1,0].set_xlabel("prod_distance")
    axs[1,1].set_xlabel("prod_distance")
    axs[2,0].scatter(time_delivery, target)
    axs[2,1].scatter(time_delivery, y_hat, color="orange")
    axs[2,0].set_xlabel("time_delivery")
    axs[2,1].set_xlabel("time_delivery")
    plt.show()

if len(sys.argv) >= 2:
    model_filename = sys.argv[1]
    model_data = pd.read_csv(model_filename)
    model_data = model_data.sort_values(by=["loss"], key = lambda s: s.astype(float))

    for i in range(2):

        thetas = np.array([float(theta)for theta in model_data.iloc[i]["thetas"].split()]).reshape(-1, 1)
        weight_order = model_data.iloc[i]["weight_order"]
        prod_distance_order = model_data.iloc[i]["prod_distance_order"]
        time_delivery_order = model_data.iloc[i]["time_delivery_order"]
        print(weight_order, prod_distance_order, time_delivery_order,model_data.iloc[i]["lambda"], " => ",model_data.iloc[i]["loss"])
        plot_comparison(weight_order,prod_distance_order,time_delivery_order, thetas, model_data.iloc[i]["lambda"])

    lambdas = [0,0.2,0.4,0.6,0.8,1.0]


    

