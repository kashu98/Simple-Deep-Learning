import numpy as np
import sys

class Activation:
    def __init__(self):
        self.X = None
        self.Y = None
        self.sign = None
    def foward(self, X):
        self.X = X

"""
All of the following functions are based on the list of https://en.wikipedia.org/wiki/Activation_function
"""

class ReLU(Activation):
    """Rectified Linear Unit 
    """
    def __init__(self):
        super().__init__()
    def foward(self, X, α=0):
        self.sign = (X <= 0)
        X[self.sign] = X[self.sign] * α
        return X
    def backward(self, dY, α=0):
        dY[self.sign] = dY[self.sign] * α
        return dY

class LReLU(ReLU):
    """Leaky Rectified Linear Unit 
    """
    def __init__(self):
        super().__init__()
    def foward(self, X):
        return super().foward(X, 0.01)
    def backward(self, dY):
        return super().backward(dY, 0.01)

class PReLU(ReLU):
    """Parameteric Rectified Linear Unit
    """
    def __init__(self):
        super().__init__()
        self.α = None
    def foward(self, X, α):
        self.α = α
        return super().foward(X, α)
    def backward(self, dY):
        return super().backward(dY, self.α)

class ELU(Activation):
    """Exponential Linear Unit
    """
    def __init__(self):
        super().__init__()
        self.α = None
    def foward(self, X, α, λ=1.0):
        self.α = α
        X = λ * X
        self.sign = (X <= 0)
        X[self.sign] = α * (np.exp(X[self.sign]) - 1.0)
        self.Y = X
        return X
    def backward(self, dY, λ=1.0):
        dY = λ * dY
        dY[self.sign] = dY[self.sign] * (self.Y + self.α)
        return dY

class SELU(ELU):
    """Scaled Exponential Linear Unit (Klambauer et al., 2017)
    """
    def __init__(self):
        super().__init__() 
        self.α = 1.67326
        self.λ = 1.0507
    def foward(self, X):
        return super().foward(X, self.α, self.λ)
    def backward(self, dY):
        return super().backward(dY, self.λ)

class Sigmoid(Activation):
    """Logistic Function
    """
    def __init__(self):
        super().__init__()
    def foward(self, X):
        self.Y = 1/(1 + np.exp(-X))
        return self.Y
    def backward(self, dY):
        dX = dY * self.Y * (1.0 - self.Y)
        return dX

class SoftPlus(Sigmoid):
    """
    """
    def __init__(self):
        super().__init__()
    def foward(self, X):
        self.X = X
        return np.log(1.0 + np.exp(X))
    def backward(self, dY):
        dX = dY * super().foward(self.X)
        return dX

class Tanh(Activation):
    """
    """
    def __init__(self):
        super().__init__()
    def foward(self, X):
        self.Y = 2.0/(1.0 + np.exp(-2 * X) - 1.0)
        return self.Y
    def backward(self, dY):
        dX = dY * (1.0 - (self.Y)**2)
        return dX

class ArcTan(Activation):
    """
    """
    def __init__(self):
        super().__init__()
    def foward(self, X):
        self.X = X
        return np.arctan(X)
    def backward(self, dY):
        dX = dY/(1.0 + (self.X)**2)
        return dX

class SoftSign(Activation):
    """
    """
    def __init__(self):
        super().__init__()
    def foward(self, X):
        self.sign = (X < 0)
        aX = X.copy()
        aX[self.sign] = -1.0 * aX[self.sign]
        self.X = aX
        return X/(1.0 + aX)
    def backward(self, dY):
        return dY/(1.0 + self.X)**2 

class Maxout:
    """
    http://blog.yusugomori.com/post/133257383300/%E6%95%B0%E5%BC%8F%E3%81%A7%E6%9B%B8%E3%81%8D%E4%B8%8B%E3%81%99-maxout-networks
    """
    def __init__(self):
        pass

def Softmax(X):
    option = X.ndim
    if option == 1:
        X = X - np.max(X)
        return np.exp(X) / np.sum(np.exp(X))
    elif option == 2:
        X = X.T
        X = X - np.max(X, axis=0)
        Y = np.exp(X) / np.sum(np.exp(X), axis=0)
        return Y.T
    else:
        sys.stderr.write('unexpected dimention data was given to Softmax function.')    