import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_


# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
# Accuracy
print("Accuracy:")
print("Mine =",accuracy_score_(y, y_hat))
print("Sklearn =",accuracy_score(y, y_hat))
# Precision
print("Precisioin:")
print("Mine =",precision_score_(y, y_hat))
print("Sklearn =",precision_score(y, y_hat))
# Recall
print("Recall:")
print("Mine =",recall_score_(y, y_hat))
print("Sklearn =",recall_score(y, y_hat))

# F1-score
print("F1 score:")
print("Mine =",f1_score_(y, y_hat))
print("Sklearn =",f1_score(y, y_hat))