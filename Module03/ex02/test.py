import numpy as np
from log_loss import log_loss_
from sklearn import metrics


# Example 1:
y1 = np.array([[0], [0]])
y_hat1 = np.array([[0], [0]])
print("Example 1:",log_loss_(y1, y_hat1))

y2 = np.array([[0],[1]])
y_hat2 = np.array([[0],[1]])
print("Example 2:",log_loss_(y2, y_hat2))

y3 = np.array([[0], [0], [0]])
y_hat3 = np.array([[1], [0], [0]])
print("Example 3:",log_loss_(y3, y_hat3))
# print(metrics.log_loss(y3, y_hat3, eps=1e-15,labels=[0,1]))


y4 =np.array([[0], [0], [0]])
y_hat4 = np.array([[1], [0], [1]])
print("Example 4:", log_loss_(y4, y_hat4))

y5 =np.array([[0], [1], [0]])
y_hat5 =np.array([[1], [0], [1]])
print("Example 5:", log_loss_(y5, y_hat5))