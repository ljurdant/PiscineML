import numpy as np
from tools import add_intercept

x = np.arange(1,6)
print(add_intercept(x))
y = np.arange(1,10).reshape((3,3))
print(add_intercept(y))
print(add_intercept(4))
print(add_intercept(np.array([])))