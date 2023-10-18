from sigmoid import sigmoid_
import numpy as np

x=np.array([0])
print("Example 1:",sigmoid_(x))
# Output:
#  array([[0.01798620996209156]])
# Example 2:
x = np.array([1])
print("Example 2:",sigmoid_(x))
# Output:
#  array([[0.8807970779778823]])
# Example 3:
x=np.array([-1])
print("Example 3:",sigmoid_(x))
# Output:
#  array([[0.01798620996209156], [0.8807970779778823], [0.5]])
x=np.array([50])
print("Example 4:",sigmoid_(x))
# Output:
#  array([[0.
x=np.array([-50])
print("Example 5:",sigmoid_(x))
# Output:
#  array([[0.