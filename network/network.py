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
        self.list_layer = []
        # List all the layers to validate the input
        self.list_layer.append(type(Convolutional()))
        self.list_layer.append(type(Pooling()))
        self.list_layer.append(type(Pading()))
        self.list_layer.append(type(Affine()))
        self.list_layer.append(type(Maxout()))
        self.list_layer.append(type(BatchNormalization()))
        self.list_layer.append(type(Dropuot()))

        self.layers = OrderedDict()
        
    def setup(self, initializer):
        pass

    def add(self, layer):
        """
        layer: layer instance below
        ___________________________________________________________________________________
        Convolutional:      Convolutional(fil, bias, stride, pad, pad_val)
        Pooling:            Pooling(ph, pw, stride, pad, pad_val)
        Affine:             Affine(weight, bias)
        Maxout:             Maxout(weight, biad)
        BatchNormalization: BatchNormalization(gamma, beta, mean_predict, varience_predict)
        Dropout:            Dropout(dropout_rate)
        ___________________________________________________________________________________
        """
        if not isinstance(layer, tuple(self.list_layer)):
            raise TypeError('The layer'+str(len(self.layers)+1) +' must be an instance of class Layer. Found: ' + str(layer))
        else:
            self.layers.update({'layer'+str(len(self.layers)+1):layer})

    def compile(self, optimizer, loss):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
    
    def gradient(self):
        pass

    def load_weight(self, file_name="weight.pkl"):
        pass
    
    def save_weight(self, file_name="weight.pkl"): #保存ファイルの拡張子はあとで考える
        pass
