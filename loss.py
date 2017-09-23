import numpy as np

"""
Xは（バッチ数、幅）からなる二次元ベクトル
Tは（バッチ数、幅）からなる二次元ベクトル
"""

def MAE(X, T):
    """Mean Absolute Error
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

def CEL(X, T):
    """Cross Entropy Loss function
    X: output of softmax function
    T: one hot vector 
    """
    return np.sum(T*np.log(np.absolute(X+10e-7)),axis=1)/X.shape[1]
