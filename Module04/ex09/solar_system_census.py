import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.poylnomial_model_extended import add_polynomial_features
from utils.zscore import zscore
from utils.data_splitter import data_spliter
from benchmark_train import classifier
from benchmark_train import predicter

def bestmodel(weight_order, height_order, bone_density_order, lambda_):
    solar_data = pd.read_csv("../data/solar_system_census.csv")
    planet_data = pd.read_csv("../data/solar_system_census_planets.csv")

    y = planet_data["Origin"]
    y = np.array(y.astype({"Origin": "int"})).reshape(-1, 1)


    weights = add_polynomial_features(zscore(np.array(solar_data["weight"])).reshape(-1,1), weight_order)
    heights = add_polynomial_features(zscore(np.array(solar_data["height"])).reshape(-1,1), height_order)
    bone_densitys = add_polynomial_features(zscore(np.array(solar_data["bone_density"])).reshape(-1,1), bone_density_order)

    x = np.concatenate((weights, heights, bone_densitys), axis=1)

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)


    thetas0 = classifier(0, x_train, y_train, lambda_)
    thetas1 = classifier(1, x_train, y_train, lambda_)
    thetas2 = classifier(2, x_train, y_train, lambda_)
    thetas3 = classifier(3, x_train, y_train, lambda_)

    y_hat = predicter(thetas0, thetas1, thetas2, thetas3, x_test)


    x0 = x_test[y_hat == 0]
    x1 = x_test[y_hat == 1]
    x2 = x_test[y_hat == 2]
    x3 = x_test[y_hat == 3]

    x0_t = x_test[y_test == 0]
    x1_t = x_test[y_test == 1]
    x2_t = x_test[y_test == 2]
    x3_t = x_test[y_test == 3]

    fig, axs = plt.subplots(3,2, sharey=True)
    axs[0,0].annotate("Predicted values",xy=(0 , 0),xytext=(0,2.5))
    axs[0,1].annotate("True values",xy=(0 , 0),xytext=(0,2.5))

    for x,x_t in zip([x0,x1,x2,x3],[x0_t,x1_t,x2_t,x3_t]):

        axs[0,0].scatter(x[:,0],x[:,1],alpha=1)
        axs[0,1].scatter(x_t[:,0],x_t[:,1],alpha=1)
       
        axs[0,0].set_ylabel("height")
        axs[0,0].set_xlabel("weight")

        axs[1,0].scatter(x[:,0], x[:,2])
        axs[1,1].scatter(x_t[:,0], x_t[:,2])

        axs[1,0].set_ylabel("bone_density")
        axs[1,0].set_xlabel("weight")


        axs[2,0].scatter(x[:,1], x[:,2])
        axs[2,1].scatter(x_t[:,1], x_t[:,2])
        axs[2,0].set_ylabel("bone_density")
        axs[2,0].set_xlabel("height")

    axs[0,0].legend(["zip=0","zip=1","zip=2","zip=3"])
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        model_filename = sys.argv[1]

        model_data = pd.read_csv(model_filename)


        fig, ax = plt.subplots(layout='constrained')

        width_const = 0.3
        multiplier = 0

        for index, row in model_data.groupby(["weight_order","height_order","bone_density_order"],as_index=False):

            width = 0
            rects = None
            for i in range(2):
                rects = ax.bar(multiplier+width,row.iloc[i]["f1score"],width=0.25, label=row.iloc[i]["lambda"])
                ax.bar_label(rects,[str(row.iloc[i]["lambda"])],padding=3)
                width+=width_const
            multiplier+=1
            rects.set_label(str(index))

        ax.set_xticks( np.arange(27) + width_const / 2, [str(index) for index,row in model_data.groupby(["weight_order","height_order","bone_density_order"])])
        ax.set_ylim(0.8, 1)
        plt.show()
 
        model_data = model_data.sort_values(by=["f1score"], key = lambda s: s.astype(float),ascending=False)

        bestmodel(model_data.iloc[0]["weight_order"],model_data.iloc[0]["height_order"],model_data.iloc[0]["bone_density_order"],model_data.iloc[0]["lambda"])        