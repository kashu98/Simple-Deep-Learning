import numpy as np

class Layer:
    def __init__(self, weight, bias):
        self.B = {'bias':bias, 'delta':None}
        self.X = {'input':None, 'output':None, 'shape':None, 'delta':None, 'batch':None, 'channel':None, 'hight':None, 'width':None}
        self.W = {'weight':weight, 'delta':None}
        if len(weight.shape) == 2:
            self.W['patch'], self.W['channel'] = 1,1
            self.W['hight'], self.W['width'] = weight.shape 
            #この場合hightが入力ノード数、widthが出力ノード数となる
        elif len(weight.shape) == 4:
            self.W['patch'], self.W['channel'], self.W['hight'], self.W['width'] = weight.shape 
    
    def foward(self, X):
        self.X['input'] = X
        self.X['shape'] = X.shape
        if len(X.shape) == 2:
            self.X['hight'], self.X['channel'] = 1,1
            self.X['batch'], self.X['width'] = X.shape
        elif len(X.shape) == 4:
            self.X['batch'], self.X['channel'], self.X['hight'], self.X['width'] = X.shape 
