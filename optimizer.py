import numpy as np

class SGD:
    """Stochastic gradient descent (SGD)
    SGD and its variants are probably the most used optimization algorithms for machine learning in general and for deep learning in particular.
    Require: Learning rate ε
    Require: Initial parameter θ.
    Algorithm: 
        while stopping criterion not met do \n
        \t    Sample a minibatch of m examples from the training set {x_1,...,x_m} with coresponding targets y_i \n
        \t    Compute gradient: g ← (1/m)*∇_θSum(L(f(x;θ),y)) \n
        \t    Apply update: θ ← θ - εg \n
        end while
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def optimize(self, parameter, gradient):
        for i in parameter:
            parameter[i]['weight'] -= self.learning_rate * gradient[i]['weight']
            parameter[i]['bias'] -= self.learning_rate * gradient[i]['bias']

class Momentum:
    """The method of momentum (Polyak, 1964)
    This method is designed to accelerate learning, especially in the face of high curvature, small but consistent gradients, or noisy gradients.
    Require: Learning rate ε, momentum parameter α.
    Require: Initial parameter θ, initial velocity v.
    Algorithm: 
        while stopping criterion not met do \n
        \t    Sample a minibatch of m examples from the training set {x_1,...,x_m} with coresponding targets y_i \n
        \t    Compute gradient: g ← (1/m)*∇_θSum(L(f(x;θ),y)) \n
        \t    Compute velocity update:  v ← αv - εg \n
        \t    Apply update: θ ← θ - v \n
        end while
    """
    def __init__(self,learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def optimize(self, parameter, gradient):
        for i in parameter:
            if not i in self.velocity.keys():
                self.velocity[i]['weight'] = np.zeros_like(parameter[i]['weight'])
                self.velocity[i]['bias'] = np.zeros_like(parameter[i]['bias'])
            self.velocity[i]['weight'] = self.momentum * self.velocity[i]['weight'] - self.learning_rate * gradient[i]['weight']
            self.velocity[i]['bias'] = self.momentum * self.velocity[i]['bias'] - self.learning_rate * gradient[i]['bias']
            parameter[i]['weight'] -= self.velocity[i]['weight']
            parameter[i]['bias'] -= self.velocity[i]['bias']
            
class AdaGrad:
    """AdaGrad algorithm (Duchi et al., 2011)
    Require: Global learning rate ε
    Require: Initial parameter θ.
    Require: Small constant δ, usually 10^-7, for numerical stability
    Algorithm: 
        Initialize gradient accumulation variable r = 0 \n
        while stopping criterion not met do \n
        \t    Sample a minibatch of m examples from the training set {x_1,...,x_m} with coresponding targets y_i \n
        \t    Compute gradient: g ← (1/m)*∇_θSum(L(f(x;θ),y)) \n
        \t    Accumulate squared gradient: r ← r + g☉g \n
        \t    Compute parameter update:  ∆θ ← -(ε/(δ +sqrt(r)))☉g \n
        \t    Apply update: θ ← θ + ∆θ \n
        end while
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.δ = 10e-7
        self.r = {}
    
    def optimize(self, parameter, gradient):
        for i in parameter:
            if not i in self.r.keys():
                self.r[i]['weight'] = np.zeros_like(gradient[i]['weight'])
                self.r[i]['bias'] = np.zeros_like(gradient[i]['bias'])
            self.r[i]['weight'] += np.square(gradient[i]['weight'])
            self.r[i]['bias'] += np.square(gradient[i]['bias'])
            parameter[i]['weight'] -= np.multiply(self.learning_rate/(self.δ + np.sqrt(self.r[i]['weight'])), gradient[i]['weight'])
            parameter[i]['bias'] -= np.multiply(self.learning_rate/(self.δ + np.sqrt(self.r[i]['bias'])), gradient[i]['bias'])
