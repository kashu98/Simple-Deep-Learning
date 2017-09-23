import numpy as np

def MAE(X, T):
    """Mean Absolute Error
    Xは（バッチ数、幅）からなる二次元ベクトル
    Tは（バッチ数、幅）からなる二次元ベクトル
    """
    return np.sum(np.absolute(X-T), axis=1)/X.shape[1]

def MSE(X, T):
    """Mean Square Error
    """
    return np.sum((X-T)**2, axis=1)/X.shape[1]

def RMSE(X, T):
    """Root Mean Square Error
    """
    return np.sqrt(np.sum((X-T)**2, axis=1)/X.shape[1])
