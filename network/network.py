from collections import OrderedDict

import sys,os
sys.path.append(os.pardir)
# import from parent direntory
from loss import *
from activation import *
from layers import *
from optimizer import *
from initializer import *

# model weights are easily stored using  HDF5 format and that the network structure can be saved in either JSON or YAML format.

class Sequential:
    '''
    ## Methods
    add: add a new layer to the neural network
    compile: add optimizer and loss function
    predict: process the input without learning (without updating the weights)
    evaluate: get the accuracy of the network
    gradient: 
    load_weight: 
    save_weight: 
    '''
    def __init__(self):
        # List all the layers to validate the input
        self.list_layer = []
        self.list_layer.append(type(Affine()))
        self.list_layer.append(type(Convolution()))
        self.list_layer.append(type(Pooling()))
        self.list_layer.append(type(Pading()))
        self.list_layer.append(type(Dropuot()))
        #self.list_layer.append(type(Maxout()))
        #self.list_layer.append(type(BatchNormalization()))
        #self.list_layer.append(type(Skip()))

        # List all the optimizer to validate the input
        self.list_opt = []
        self.list_opt.append(type(SGD()))
        self.list_opt.append(type(Momentum()))
        #self.list_opt.append(type(Nesterov_Momentum()))
        self.list_opt.append(type(AdaGrad()))
        #self.list_opt.append(type(RMSprop()))
        #self.list_opt.append(type(Adam()))
        #self.list_opt.append(type(Adamdelta()))
        #self.list_opt.append(type(AdaMax()))
        #self.list_opt.append(type(Nadam()))

        self.layers = OrderedDict()
        self.params = {}
        self.grads = {}
        self.opt = None

    def add(self, layer):
        if not isinstance(layer, tuple(self.list_layer)):
            raise TypeError('The layer' +str(len(self.layers)) +' must be a layer class definened in layers.py. Not found: ' +str(layer))
        else:
            ly = layer
            self.layers.update({str(len(self.layers)):{'layer':ly, 'X':None, 'dY':None}})

    def compile(self, optimizer, loss):
        if not isinstance(optimizer, tuple(self.list_opt)):
            raise TypeError('The optimizer ' +str(optimizer)) + ' is not defined.')
        else:
            self.opt = optimizer

        #交差エントロピー誤差が入力されれば、出力層に「ソフトマックス関数」を設定
        #二乗和誤差MSEが入力されれば、出力層に恒等関数を設定する

    def predict(self, X):
        self.layers['0']['X'] = X
        for i in range(len(self.layers)):
            self.layers[str(i+1)]['X'] = self.layers[str(i)]['layer'].forward(self.layers[str(i)]['X'])
        return self.layers[str(len(self.layers)-1)]['X']

        #ニューラルネットワークの推論で答えを一つだけ出力する場合は、スコアの最大値のみが必要なので、Softmaxレイヤは不必要
    
    def train(self, X, T):
        # forward
        self.layers['0']['X'] = X
        for i in range(len(self.layers)-1):
            self.layers[str(i+1)]['X'] = self.layers[str(i)]['layer'].forward(self.layers[str(i)]['X'])
        self.layers[str(len(self.layers)-1)]['layer'].forward(self.layers[str(len(self.layers)-1)]['X'])
        
        # get all the parameters
        for i in self.layers:
            if self.layers[i]['layer'].has_params() == True:
                self.params[i] = self.layers[i]['layer'].get_params()
        
        #最終的に得られる誤差をバッチ数で割って正規化してから逆伝播するように修正する
        self.layers[str(len(self.layers)-1)]['dY'] = T
        for i in range(len(self.layers)-1):
            self.layers[str(len(self.layers)-i-2)]['dY'] = self.layers[str(len(self.layers)-i-1)]['layer'].backward(self.layers[str(len(self.layers)-i-1)]['dY'])
        self.layers['0']['layer'].backward(self.layers['0']['dY'])

        # get all the gradient
        for i in self.layers:
            if self.layers[i]['layer'].has_params() == True:
                self.grads[i] = self.layers[i]['layer'].get_grads()

        self.opt.optimize(self.params, self.grads)
    
    def evaluate(self, X, T):
        pass
    
    def load_weight(self, filename="weight.hdf5"):
        file = h5.File(filename, 'r')
        # 以下は例
        weight = np.array(f['weight'])
        bias = np.array(f['bias'])
    
    def save_weight(self, filename="weight.hdf5"):
        file = h5.File(filename, 'w')
        # layerにweightがあるか確かめる
