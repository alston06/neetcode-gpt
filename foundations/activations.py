import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        sig = 1 / (1 + np.exp(-z))
        return np.round(sig, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        res = [0] * len(z)
        for i, n in enumerate(z):
            res[i] = max(0.0, n)
        
        return res
