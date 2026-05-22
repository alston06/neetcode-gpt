import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:

        z = z - np.max(z)

        summation = 0
        for i in z:
            summation += np.exp(i)

        res = [0] * len(z)

        for i, n in enumerate(z):
            res[i] = round((np.exp(n)/summation), 4)

        return res