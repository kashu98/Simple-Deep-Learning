from collections import OrderedDict

import sys,os
sys.path.append(os.pardir)
# import from parent direntory
from loss import *
from activation import *
from layers import *
from optimizer import *
from initializer import *

class Sequential:
    """
    """
    def __init__(self):
        self.list = []
        self.list.append(type(ReLU()))
        self.list.append(type(LReLU()))
        self.list.append(type(PReLU()))
        self.list.append(type(ELU()))
        self.list.append(type(SELU()))
        self.list.append(type(Sigmoid()))
        self.list.append(type(SoftPlus()))
        self.list.append(type(SoftSign()))
        self.list.append(type(Convolutional()))
        self.list.append(type(Pooling()))
        self.list.append(type(Affine()))
        self.list.append(type(Maxout()))
        self.list.append(type(BatchNormalization()))
        self.list.append(type(Dropuot()))

        self.layers = OrderedDict()
        self.params = 
        
    def generate(self, initializer, weight, bias=None):
        pass

    def add(self, layer):
        """
        layer: layer instance below
        ========================================================================================
        Convolutional:      Convolutional(fil, bias, stride, pad, pad_val)
        Pooling:            Pooling(ph, pw, stride, pad, pad_val)
        Affine:             Affine(weight, bias)
        Maxout:             Maxout(weight, biad)
        BatchNormalization: BatchNormalization(gamma, beta, mean_predict, varience_predict)
        Dropout:            Dropout(dropout_rate)
        + Activation functions
        """
        if not isinstance(layer, tuple(self.list)):
            raise TypeError('The layer'+str(len(self.layers)+1) +' must be an instance of class Layer. Found: ' + str(layer))
        self.layers.update({'layer'+str(len(self.layers)+1):layer})
