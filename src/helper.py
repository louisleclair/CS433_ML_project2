import numpy as np
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================
# Cost functions
# ==============================================

def MSE(prediction, target):
    """Compute the mean square error, adapt for tensors.

    Args:
        prediction (tensor): predicted data
        target (tensor): target data

    Returns:
        float: the MSE between the prediction and target data.
    """
    pred = prediction.detach().numpy()
    tar = target.numpy()
    return sklearn.metrics.mean_squared_error(y_true=tar, y_pred=pred)

def MAE(prediction, target):
    """Compute the mean absolute error, adapt for tensors

    Args:
        prediction (tensor): predicted data
        target (tensor): target data

    Returns:
        float: the MAE between the prediction and target data. 
    """
    pred = prediction.detach().numpy()
    tar = target.numpy()
    return sklearn.metrics.mean_absolute_error(y_true=tar, y_pred=pred)

# ==============================================
# Cross Correlation
# ==============================================

# Cross Correlation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k):
    """Create a random division of the data between training and test set depending of a list of random indices and an index.

    Args:
        y (np.array): Nx1 array of the target data.
        x (np.array): NxD array of the feature data.
        k_indices (np.array): MxP array of random indices used to divide the data in test and training set.
        k (int): k index where to choose the random array of indices in k_indices.

    Returns:
        tuple: division of the data in training and test set.
    """
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    return x_tr, x_te, y_tr, y_te

def cross_correlation_regression(y, x, k_indices, k_fold, f, f_error):
    """Compute and print the loss for different k depending of the number of k_fold for different training and test set.
    Show the coefficients of the weights.

    Args: 
        y (np.array): Nx1 array of the target data.
        x (np.array): NxD array of the feature data.
        k_indices (np.array): MxP array of random indices used to divide the data in test and training set.
        k_fold (int): number of set, we are going to divide our dataset.
        f (function): function we use to do linear regression.
        f_error (function): error function use to compute the loss between our predicted and target data.
    """
    for k in range(k_fold): 
        x_tr, x_te, y_tr, y_te = cross_validation(y, x, k_indices, k)
        regr = f
        regr.fit(x_tr, y_tr)
        pred = regr.predict(x_te)
        print("k = {} and loss= {:.6f} and coef= {}".format(k+1, f_error(y_true=y_te, y_pred=pred), np.round(regr.coef_, 4)))

def cross_correlation_boost(y, x, k_indices, k_fold, f, f_error):
    """Compute and print the loss for different k depending of the number of k_fold for different training and test set.
    Almost the same method as cross_correlation_regression but print the coefficients of the weights and the biais, to use with XGBoost.

    Args:
        y (np.array): Nx1 array of the target data.
        x (np.array): NxD array of the feature data.
        k_indices (np.array): MxP array of random indices used to divide the data in test and training set.
        k_fold (int): number of set, we are going to divide our dataset.
        f (function): function we use to do linear regression.
        f_error (function): error function use to compute the loss between our predicted and target data.
    """
    for k in range(k_fold):
        x_tr, x_te, y_tr, y_te = cross_validation(y, x, k_indices, k)
        regr = f
        regr.fit(x_tr, y_tr)
        pred = regr.predict(x_te)
        coefficient = regr.coef_
        inter = regr.intercept_
        print("k = {}, Loss= {:.6f}, intercept= {}, coefficients= {}".format(k+1, f_error(y_true=y_te, y_pred=pred),np.round(inter,4),np.round(coefficient,4)))

# ==============================================
# Data analysis
# ==============================================

def show_heatmap(data):
    """Show the correlation map between the each row of the given dataset."""
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

# ==============================================
# Neural network
# ==============================================

def train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs, f, device):
    """Training method for our neural network, train the model on the training set and test it on the test set. 
    Print the loss average of the loss of the given test set.
    Inspire from lab10 of the class.

    Args:
        model : Neural network model.
        criterion : loss function use in our model.
        dataset_train (tensor): training set.
        dataset_test (tensor): test set.
        optimizer : optimezer used.
        num_epochs (int): number of iterations.
        f (function): loss function used to compute the difference between the prediction and the target.
        device : gpu if available otherwise cpu.
    """
    print("Starting training")
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        # Test the quality on the test set
        model.eval()
        accuracies_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies_test.append(f(prediction, batch_y))

        if 'MAE' in str(f):
            print("Epoch {} | Test MAE: {:.5f}".format(epoch, sum(accuracies_test).item()/len(accuracies_test)))
        elif 'MSE' in str(f):
            print("Epoch {} | Test MSE: {:.5f}".format(epoch, sum(accuracies_test).item()/len(accuracies_test)))

