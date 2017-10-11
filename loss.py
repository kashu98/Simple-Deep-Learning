import numpy as np

def MAE(X, T):
    """Mean Absolute Error
    
    ## Output
        Loss of the whole mini-batch (one value will be returned)
    """
    return np.sum(np.absolute(X-T))/X.shape[0]

def MSE(X, T):
    """Mean Square Error

    ## Output
        Loss of the whole mini-batch (one value will be returned)
    """
    return np.sum((X-T)**2)/X.shape[0]

def RMSE(X, T):
    """Root Mean Square Error

    ## Output
        Loss of the whole mini-batch (one value will be returned)
    """
    return np.sqrt(np.sum((X-T)**2)/X.shape[0])

def CEL(X, T):
    """Cross Entropy Loss
    X: output of softmax function
    T: training data in the form of one-hot vector 
    
    ## Output
        Loss of the whole mini-batch (one value will be returned)
    """
    return -np.sum(T*np.log(np.absolute(X+10e-7)))/X.shape[0]
