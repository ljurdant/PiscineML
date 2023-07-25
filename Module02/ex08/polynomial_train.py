import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregressionmulti import MyLinearRegressionMulti
from polynomial_model import add_polynomial_features

data = pd.read_csv("../data/are_blue_pills_magics.csv")

x = np.array(data["Micrograms"]).reshape(-1, 1)
y = np.array(data["Score"]).reshape(-1, 1)

x_1 = x
x_2 = add_polynomial_features(x, 2)
x_3 = add_polynomial_features(x, 3)
x_4 = add_polynomial_features(x, 4)
x_5 = add_polynomial_features(x, 5)
x_6 = add_polynomial_features(x, 6)

theta1 = np.array([[89.0], [-8]])
theta2 = np.array([[89.0], [-8], [2]])
theta3 = np.array([[89], [-8], [-1], [2]])
theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]])
theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]])
theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]])

continuous_x = np.arange(1,7.01, 0.01).reshape(-1,1)

lr = MyLinearRegressionMulti(theta1)
lr.fit_(x_1, y)
y1 = lr.predict_(x_1)
print("MSE 1 => ", lr.mse_(x_1, y))
y1_continuous = lr.predict_(continuous_x)

lr.thetas = theta2
lr.fit_(x_2, y)
y2 = lr.predict_(x_2)
print("MSE 2 => ", lr.mse_(x_2, y))
y2_continuous = lr.predict_(add_polynomial_features(continuous_x, 2))

lr.alpha = 1e-5
lr.thetas = theta3
lr.fit_(x_3, y)
y3 = lr.predict_(x_3)
print("MSE 3 => ", lr.mse_(x_3, y))
y3_continuous = lr.predict_(add_polynomial_features(continuous_x, 3))

lr.alpha = 1e-6
lr.max_iter = 500000
lr.thetas = theta4
lr.fit_(x_4, y)
y4 = lr.predict_(x_4)
print("MSE 4 => ", lr.mse_(x_4, y))
y4_continuous = lr.predict_(add_polynomial_features(continuous_x, 4))


lr.alpha = 1e-8
lr.thetas = theta5
lr.fit_(x_5, y)
y5 = lr.predict_(x_5)
print("MSE 5 => ", lr.mse_(x_5, y))
y5_continuous = lr.predict_(add_polynomial_features(continuous_x, 5))


lr.alpha = 1e-9
lr.thetas = theta6
lr.fit_(x_6, y)
y6 = lr.predict_(x_6)
print("MSE 6 => ", lr.mse_(x_6, y))
y6_continuous = lr.predict_(add_polynomial_features(continuous_x, 6))


plt.plot(x, y, 'o')
plt.plot(continuous_x, y1_continuous)
plt.plot(continuous_x, y2_continuous)
plt.plot(continuous_x, y3_continuous)
plt.plot(continuous_x, y4_continuous)
plt.plot(continuous_x, y5_continuous)
plt.plot(continuous_x, y6_continuous)
plt.xlabel("Micrograms")
plt.ylabel("Score")

plt.legend(
    ["Data",
     "y1",
     "y2",
     "y3",
     "y4",
     "y5",
     "y6"],
     bbox_to_anchor=(0,1.1), loc="upper left", ncol=4, edgecolor="white")


plt.show()