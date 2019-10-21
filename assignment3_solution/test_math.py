import numpy as np
from numpy import linalg as LA
a = np.array([1,2,3])
b = np.array([-1,1,4])
# c = np.array([[ 1, 2, 3]-[-1, 1, 4]])
print(LA.norm(a-b))
c= np.array([2,1,-1])
print(LA.norm(c))