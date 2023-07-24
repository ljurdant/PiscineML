import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregressionmulti import MyLinearRegressionMulti
from mylinearregressionuni import MyLinearRegressionUni


def plot(x, y, ymodel, xlabel, ylabel="sell price"):
    plt.plot(x, y, 'o')
    plt.plot(x, ymodel, 'o')
    plt.legend(["Data","Prediction"],bbox_to_anchor=(0,1.1), loc="upper left", ncol=2, edgecolor="white")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

#Parsing
data = pd.read_csv("../data/spacecraft_data.csv")

age = np.array(data["Age"]).reshape(-1,1)
thrust = np.array(data["Thrust_power"]).reshape(-1,1)
terameters = np.array(data["Terameters"]).reshape(-1,1)
sell_price = np.array(data["Sell_price"]).reshape(-1,1)

#Age
thetas = np.array([[1000.0], [-1.0]])
alpha = 2.5e-4
max_iter = 100000
lRUni = MyLinearRegressionUni(thetas, alpha, max_iter)
lRUni.fit_(age, sell_price)
ageModel = lRUni.predict_(age)
# plot(age, sell_price, ageModel, "age")
print("Age mse: ", lRUni.mse_(sell_price, ageModel))

# #Thrust
lRUni.thetas = np.array([[1.0], [7.0]])
lRUni.max_iter = 100000
lRUni.alpha = 0.0001
lRUni.fit_(thrust, sell_price)
thrustModel = lRUni.predict_(thrust)
# plot(thrust, sell_price, thrustModel, "thrust")
print("Thrust mse: ", lRUni.mse_(sell_price, thrustModel))

# #Terameters
lRUni.thetas = np.array([[1.0], [-2]])
lRUni.max_iter = 100000
lRUni.alpha = 0.0001
lRUni.fit_(terameters, sell_price)
terametersModel = lRUni.predict_(terameters)
plot(terameters, sell_price, terametersModel, "terameters")
print("Terameters mse: ", lRUni.mse_(sell_price, terametersModel))

#Multivariate
Xdata = np.array(data[["Age","Thrust_power","Terameters"]])
lRMulti = MyLinearRegressionMulti([[1.0], [1.0], [1.0], [1.0]], 1e-4, 600000)
lRMulti.fit_(Xdata, sell_price)
print("Multi thetas = ", lRMulti.thetas)
multiModel = lRMulti.predict_(Xdata)
plot(age, sell_price, multiModel, "age")
