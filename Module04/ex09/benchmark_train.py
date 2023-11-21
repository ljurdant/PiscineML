import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import threading
from datetime import datetime


from utils.my_logistic_regression import MyLogisticRegression
from utils.data_splitter import data_spliter
from utils.other_metrics import f1_score_
from utils.poylnomial_model_extended import add_polynomial_features
from utils.zscore import zscore


def classifier(zipcode, x_train, y_train, lambda_):

    thetas = np.concatenate(([10], np.random.uniform(0,1,x_train.shape[1]) )).reshape(-1, 1)
    y_train = y_train == zipcode

    max_iter = 10000
    myLr = MyLogisticRegression(thetas, max_iter=max_iter, lambda_=lambda_)
    return myLr.fit_(x_train, y_train.reshape(-1, 1))

def predicter(thetas1, thetas2, thetas3, thetas4, x_test_):
    myLr1 = MyLogisticRegression(thetas1)
    y1 = myLr1.predict_(x_test_)

    myLr2 = MyLogisticRegression(thetas2)
    y2 = myLr2.predict_(x_test_)
    
    myLr3 = MyLogisticRegression(thetas3)
    y3 = myLr3.predict_(x_test_)

    myLr4 = MyLogisticRegression(thetas4)
    y4 = myLr4.predict_(x_test_)

    y_all = np.concatenate((y1,y2,y3,y4), axis=1)
    return np.argmax(y_all, axis=1)

def thread_function(weight_order, height_order, bone_density_order, lambda_):

    x_train_ = np.concatenate((x_train[:,:weight_order],x_train[:,3:3+height_order],x_train[:,6:6+bone_density_order]), axis=1)
    x_test_ = np.concatenate((x_test[:,:weight_order],x_test[:,3:3+height_order],x_test[:,6:6+bone_density_order]), axis=1)

    lock.acquire()
    print(f"Starting {weight_order} {height_order} {bone_density_order} {lambda_}")
    lock.release()

    thetas0 = classifier(0, x_train_, y_train, lambda_)
    thetas1 = classifier(1, x_train_, y_train, lambda_)
    thetas2 = classifier(2, x_train_, y_train, lambda_)
    thetas3 = classifier(3, x_train_, y_train, lambda_)

    y_hat = predicter(thetas0, thetas1, thetas2, thetas3, x_test_)

    f1_score = [f1_score_(y_test, y_hat,pos_label=i) for i in range(4)]

    lock.acquire()
    print(f"{weight_order} {height_order} {bone_density_order} {lambda_}: {'|'.join(str(f1) for f1 in f1_score)}")
    lock.release()

    models.append([f1_score, weight_order, height_order, bone_density_order, lambda_])

if __name__ == '__main__':


    solar_data = pd.read_csv("../data/solar_system_census.csv")
    planet_data = pd.read_csv("../data/solar_system_census_planets.csv")

    y = planet_data["Origin"]
    y = np.array(y.astype({"Origin": "int"})).reshape(-1, 1)


    weights = add_polynomial_features(zscore(np.array(solar_data["weight"])).reshape(-1,1), 3)
    heights = add_polynomial_features(zscore(np.array(solar_data["height"])).reshape(-1,1), 3)
    bone_densitys = add_polynomial_features(zscore(np.array(solar_data["bone_density"])).reshape(-1,1), 3)

    x = np.concatenate((weights, heights, bone_densitys), axis=1)

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)

    lock = threading.Lock()
    models = []
    threads = []

    for weight_order in range(1, 4):
        for height_order in range(1,4):
            for bone_density_order in range(1,4):
                for lambda_ in [0.0, 1.0]:
                    thread = threading.Thread(target=thread_function, args=(weight_order, height_order, bone_density_order, lambda_))
                    threads.append(thread)
                    thread.start()

    for thread in threads:
        thread.join()

    now = datetime.now()
    filename = "models/model_"+now.strftime("%m_%d_%H:%M:%S")+".csv"

    f = open(filename, "x")
    print("f1score,f1scores,weight_order,height_order,bone_density_order,lambda", file=f)
    f.close()

    models.sort(key= lambda x: (x[1],x[2],x[3],x[4]))
    for model in models:
        [f1score, weight_order, height_order, bone_density_order, lambda_] = model
        with open(filename, "a") as file:
            print(np.mean(f1score),"|".join(str(round(score, 3)) for score in f1score), weight_order, height_order, bone_density_order,lambda_,sep=",",file=file)