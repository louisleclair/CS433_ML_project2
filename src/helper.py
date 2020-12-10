import numpy as np
import sklearn

def MSE(prediction, target):
    pred = prediction.detach().numpy()
    tar = target.numpy()
    return sklearn.metrics.mean_squared_error(y_true=tar, y_pred=pred)

def MAE(prediction, target):
    pred = prediction.detach().numpy()
    tar = target.numpy()
    return sklearn.metrics.mean_absolute_error(y_true=tar, y_pred=pred)

# Cross validation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    return x_tr, x_te, y_tr, y_te