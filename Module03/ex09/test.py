import numpy as np
from sklearn.metrics import confusion_matrix
from confusion_matrix import confusion_matrix_
y_hat = np.array([["norminet"], ["dog"], ["norminet"], ["norminet"], ["dog"], ["bird"]])
y = np.array([["dog"], ["dog"], ["norminet"], ["norminet"], ["dog"], ["norminet"]])
# Example 1:
print("Mine =",confusion_matrix_(y, y_hat))
print("Sklearn =",confusion_matrix(y, y_hat))
## Output:

# Example 2:
## your implementation
print("Mine =",confusion_matrix_(y, y_hat, labels=["dog", "norminet"]))

## sklearn implementation
print("Sklearn =",confusion_matrix(y, y_hat, labels=["dog", "norminet"]))

print("Df option :")
print(confusion_matrix_(y, y_hat, df_option=True))
print(confusion_matrix_(y, y_hat, labels=["bird", "dog"], df_option=True))
print()
print("Correction tests:")
print("Example 1:")
print(confusion_matrix_(np.array(['a', 'b', 'c']), np.array(['a', 'b', 'c'])))
print()

print("Example 2:")
print(confusion_matrix_(np.array(['a', 'b', 'c']), np.array(['c', 'a', 'b'])))
print()

print("Example 3:")
print(confusion_matrix_(np.array(['a', 'a', 'a']), np.array(['a', 'a', 'a'])))
print()

print("Example 4:")
print(confusion_matrix_(np.array(['a', 'a', 'a']), np.array(['a', 'a', 'a']), labels=[]))
