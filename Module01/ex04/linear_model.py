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
plt.grid()
plt.show()

step = 2
size = 6
legends = []
count = 0
for t0 in range(89-step*size // 2,89+step*size // 2, step):
    t1s = np.arange(-14, -4, 0.1)
    J = []
    for t1 in t1s:
        lr = MyLR(np.array([t0, t1]))
        J.append(lr.loss_(Yscore, lr.predict_(Xpill)))
    color = str(0 + count/size)
    plt.plot(t1s, J, linestyle="solid", color=color)
    legends.append("J(θ0="+str(t0)+",θ1)")
    count+=1
ax = plt.gca()
ax.set_ylim([15, 145])
plt.xlabel("θ1")
plt.ylabel("cost function J(θ0, θ1)")
plt.legend(legends, loc="lower right")
plt.grid()
plt.show()

def mse_(y, y_hat):
    return np.mean((y_hat - y)**2)

print(mse_(Yscore, Y_model1))