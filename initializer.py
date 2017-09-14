import numpy as np
import math

class Distribution():
    def uniform(self, shape, low=0.0, high=1.0):
        return np.random.uniform(low, high, shape)
    def normal(self, shape, ave=0.0, stdev=1.0):
        return np.random.normal(ave, stdev, shape)
    def beta(self, shape, a=2.0, b=2.0):
        return np.random.beta(a, b, shape)

class WeightInitializer(Distribution):
    def __init__(self, X, Y):
        """
        X: Output of the former node. X[batch_num, componet]
        Y: Input for the next node. Y[batch_num, componet]
        """
        self.ilen = len(X[0])
        self.olen = len(Y[0])
        self.shape = [self.ilen, self.olen]
    def Xavier_uniform(self):
        high = math.sqrt(6.0/(self.ilen+self.olen))
        low = - high
        return super().uniform(self.shape, low, high)
    def Xavier_simple(self, ave=0.5):
        var = 1.0/self.ilen
        return super().normal(self.shape, ave, math.sqrt(var))
    def Xavier_normal(self, ave=0.5):
        var = 2.0/(self.ilen + self.olen)
        return super().normal(self.shape, ave, math.sqrt(var))
    def He_simple(self, ave=0.5):
        var = 2.0/self.ilen
        return super().normal(self.shape, ave, math.sqrt(var))
    def He_normal(self, ave=0.5):
        var = 6.0/(self.ilen + self.olen)
        return super().normal(self.shape, ave, math.sqrt(var))
    def zero(self):
        return np.zeros(self.shape)
    def one(self):
        return np.ones(self.shape)

class FilterInitializer(Distribution):
    def __init__(self, num, channel=1, hight=3, width=3):
        self.num = num
        self.channel = channel
        self.hight = hight
        self.width = width
        self.shape = [self.num, self.channel, self.hight, self.width]
    def __call__(self, X):
        self.channel = len(X[0])
    def normal(self):
        return super().normal(self.shape)
    def zero(self):
        return np.zeros(self.shape)
