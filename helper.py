import numpy as np

def MSE(prediction, target):
    n = target.shape[0]
    return 1/(2*n) * np.mean((prediction-target)**2)