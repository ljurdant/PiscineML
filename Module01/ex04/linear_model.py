import sys, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.abspath(__file__))[:-4]+"ex03"
sys.path.append(dirname)


from my_linear_regression import MyLinearRegression as MyLR

data = pd.read_csv("../data/are_blue_pills_magic.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model1.fit_(Xpill, Yscore)
Y_model1 = linear_model1.predict_(Xpill)
plt.plot(Xpill, Yscore, 'o')
plt.plot(Xpill, Y_model1, linestyle="dashed", marker='s')
plt.legend(["Spredict(pills)","Strue(pills)"],bbox_to_anchor=(0,1.1), loc="upper left", ncol=2, edgecolor="white")
plt.xlabel("Quantity of blue pills (in micrograms)")
plt.ylabel("Space driving score")
plt.show()