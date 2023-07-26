import numpy as np
from data_spliter import data_spliter

x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
# Example 1:
print(data_spliter(x1, y, 0.8))

print(data_spliter(x1, y, 0.5))
# Output:
x2 = np.array([[ 1, 42], [300, 10],[ 59, 1], [300, 59],[ 10, 42]])
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
# Example 3:
print(data_spliter(x2, y, 0.8))
# Output:

# Example 4:
print(data_spliter(x2, y, 0.5))
# Output:

#Example 5
print(data_spliter(x2, y, 0))
print(data_spliter(x2, y, 1))
print(data_spliter([], [], 1))