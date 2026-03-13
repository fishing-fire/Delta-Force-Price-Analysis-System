import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

# 标准差
def STD(pred, true):
    return np.std(pred - true)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def metric_per_channel(pred, true):
    mae = np.mean(np.abs(pred - true), axis=(0, 1))
    mse = np.mean((pred - true) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true), axis=(0, 1))
    mspe = np.mean(np.square((pred - true) / true), axis=(0, 1))

    return mae, mse, rmse, mape, mspe
