import activation as af
import numpy as np
"""
Reference: http://www.deeplearningbook.org/contents/optimization.html
"""

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
    def __init__(self, ε=0.01):
        self.ε = ε
    def __call__(self,):
        pass

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
    def __init__(self,):
        pass
    def __call__(self,):
        pass

class Nesterov_Momentum(Momentum):
    """(Sutskever, 2013)
    Sutskever introduced a variant of the momentum algorithm that was inspired by Nesterov’s accelerated gradient method.
    Require: Learning rate ε, momentum parameter α.
    Require: Initial parameter θ, initial velocity v.
    Algorithm: 
        while stopping criterion not met do \n
        \t    Sample a minibatch of m examples from the training set {x_1,...,x_m} with coresponding targets y_i \n
        \t    Apply interim update: θ* ← θ + αv \n
        \t    Compute gradient: g ← (1/m)*∇_θSum(L(f(x;θ*),y)) \n
        \t    Compute velocity update:  v ← αv - εg \n
        \t    Apply update: θ ← θ - v \n
        end while
    """
    def __init__(self,):
        pass
    def __call__(self,):
        pass

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
    def __init__(self,):
        pass
    def __call__(self,):
        pass

class RMSProp:
    """The RMSProp algorithm (Hinton, 2012)
    This algorithm modiﬁes AdaGrad to perform better in the non-convex setting by changing the gradient accumulation into an exponentially weighted moving average.
    Require: Global learning rate ε, decay rate ρ.
    Require: Initial parameter θ.
    Require: Small constant δ, usually 10^-6, used to stabilize divition by small numbers
    Algorithm: 
        Initialize accumulation variables γ = 0 \n
        while stopping criterion not met do \n
        \t    Sample a minibatch of m examples from the training set {x_1,...,x_m} with coresponding targets y_i \n
        \t    Compute gradient: g ← (1/m)*∇_θSum(L(f(x;θ),y)) \n
        \t    Accumulate squared gradient: γ ← ργ + (1-ρ)g☉g \n
        \t    Compute parameter update:  ∆θ ← -(ε/sqrt(δ+γ))☉g \n
        \t    Apply update: θ ← θ + ∆θ \n
        end while
    """
    def __init__(self,):
        pass
    def __call__(self,):
        pass

class Adam:
    """Adam (Kingma and Ba, 2014)
    The name “Adam” derives from the phrase “adaptive moments.”
    Require: Step size ε (Sugested default: 0.001).
    Require: Exponential decay rates for moment estimates, ρ_1 and ρ_2 in [0,1) (Sugested defaults: 0.9 and 0.999 respectively).
    Require: Initial parameter θ.
    Require: Small constant δ used for numerical stabilization (Sugested default: 10^-8).
    Algorithm: 
        Initialize 1st and 2nd moment variables s = 0, r = 0 \n
        Initialize time step t = 0 \n
        while stopping criterion not met do \n
        \t    Sample a minibatch of m examples from the training set {x_1,...,x_m} with coresponding targets y_i \n
        \t    Compute gradient: g ← (1/m)*∇_θSum(L(f(x;θ),y)) t ← t + 1 \n
        \t    Update biased ﬁrst moment estimate: s ← ρ_1*s + (1-ρ_1)g \n
        \t    Update biased second moment estimate: r ← ρ_2*r + (1-ρ_2)g☉g \n
        \t    Correct bias in ﬁrst moment: s ← s/(1-ρ_1^t) \n
        \t    Correct bias in second moment: t ← t/(1-ρ_2^t) \n
        \t    Compute parameter update:  ∆θ = -(ε*s/sqrt(r)+δ)☉g \n
        \t    Apply update: θ ← θ + ∆θ \n
        end while
    """
    def __init__(self,):
        pass
    def __call__(self,):
        pass
