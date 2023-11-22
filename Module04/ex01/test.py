from l2_reg import l2, iterative_l2

import numpy as np

# Example 1:
print(iterative_l2(np.ones(10).reshape(-1, 1)))
# Output:
911.0
print(l2(np.ones(10).reshape(-1, 1)))
# Output:
911.0
# Example 3:
print(iterative_l2(np.arange(1, 10).reshape(-1, 1)))
# Output:
36.25
# Example 4:
print(l2(np.arange(1, 10).reshape(-1, 1)))
# Output

# Example 3:
print(iterative_l2(np.array([50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]).reshape(-1, 1)))
# Output:
36.25
# Example 4:
print(l2(np.array([50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]).reshape(-1, 1)))
# Output