import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def mse(targets, preds):
    targets = np.array(targets).flatten()
    preds = np.array(preds).flatten()
    return mean_squared_error(targets, preds)

def mae(targets, preds):
    targets = np.array(targets).flatten()
    preds = np.array(preds).flatten()
    return mean_absolute_error(targets, preds)

def mape(targets, preds):
    targets = np.array(targets).flatten()
    preds = np.array(preds).flatten()
    return mean_absolute_percentage_error(targets, preds) * 100

def rmse(targets, preds):
    return np.sqrt(mse(targets, preds))

def evaluate_all(targets, preds):
    return {
        'MSE': mse(targets, preds),
        'RMSE': rmse(targets, preds),
        'MAE': mae(targets, preds),
        'MAPE': mape(targets, preds)
    }