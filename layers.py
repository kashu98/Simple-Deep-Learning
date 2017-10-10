import numpy as np
from collections import OrderedDict
from initializer import *
from activation import *
# ネットワークのサイズを指定されれば自動でweightとbiasを生成するモデルに変える

class Layer2D:
    '''Model for fully conected layers
    '''
    def __init__(self, layer_size, activation=ReLU()):
        # List all the activation functions to validate the input
        self.list_LU = []
        self.list_S = []
        # Initialize with He=====================================
        self.list_LU.append(type(Identity()))
        self.list_LU.append(type(ReLU()))
        self.list_LU.append(type(LReLU()))
        self.list_LU.append(type(PReLU()))
        self.list_LU.append(type(ELU()))
        self.list_LU.append(type(SELU()))
        self.list_LU.append(type(SoftPlus()))
        # Initialize with Xavier=================================
        self.list_S.append(type(Sigmoid()))
        self.list_S.append(type(Tanh()))
        self.list_S.append(type(ArcTan()))
        self.list_S.append(type(SoftSign()))
        # No initializer is needed==============================
        
        if not isinstance(activation, (tuple(self.list_LU), tuple(self.list_S))):
            raise TypeError('The activation function '+ str(activation) + ' is not defined in the activation.py.')
        else:
            self.act = activation
        
        self.X = {'input':None, 'output':None, 'shape':None, 'delta':None, 'batch':None, 'width':None}
        self.W = {'weight':None, 'delta':None, 'shape':None, 'hight':None, 'width':layer_size}
        # この場合hightが入力ノード数、widthが出力ノード数となる
        self.B = {'bias':None, 'delta':None, 'shape':(1, layer_size), 'hight':1, 'width':layer_size}

    def forward(self, X):
        self.X['input'] = X
        self.X['shape'] = X.shape
        self.X['batch'], self.X['width'] = X.shape
        self.W['hight'] = self.X['width']
        self.W['shape'] = (self.W['hight'], self.W['width'])
        #初めてXが渡されたときにのみ重みを初期化する
        if self.W['weight'] is None:
            init = WeightInitializer(self.W['shape'])
            if isinstance(self.act, tuple(self.list_LU)):
                self.W['weight'] = init.He_normal()
                # He_simple に変えれば少しの精度を犠牲に処理速度の向上が見込める
            else:
                self.W['weight'] = init.Xavier_normal()
        if self.B['bias'] is None:
            init = WeightInitializer(self.B['shape'])
            self.B['bias'] = init.ones()

    def get_output_shape(self):
        pass

class Layer3D:
    '''Model for 3D layer
    '''
    def __init__(self, patch_size=None, kernel_size=(None,None), activation=ReLU()):
        # List all the activation functions to validate the input
        self.list_LU = []
        self.list_S = []
        # Initialize with He=====================================
        self.list_LU.append(type(Identity()))
        self.list_LU.append(type(ReLU()))
        self.list_LU.append(type(LReLU()))
        self.list_LU.append(type(PReLU()))
        self.list_LU.append(type(ELU()))
        self.list_LU.append(type(SELU()))
        self.list_LU.append(type(SoftPlus()))
        # Initialize with Xavier=================================
        self.list_S.append(type(Sigmoid()))
        self.list_S.append(type(Tanh()))
        self.list_S.append(type(ArcTan()))
        self.list_S.append(type(SoftSign()))
        if not isinstance(activation, (tuple(self.list_LU), tuple(self.list_S))):
            raise TypeError('The activation function '+ str(activation) + ' is not defined in the activation.py.')
        else:
            self.act = activation

        self.X = {'input':None, 'output':None, 'shape':None, 'delta':None, 'batch':None, 'channel':None, 'hight':None, 'width':None}
        self.W = {'weight':None, 'delta':None, 'shape':None, 'patch':patch_size, 'channel':None, 'hight':kernel_size[0], 'width':kernel_size[1]}
        self.B = {'bias':None, 'delta':None, 'shape':(patch_size,1,1), 'patch':patch_size, 'hight':1, 'width':1}
        
    def forward(self, X):
        self.X['input'] = X
        self.X['shape'] = X.shape
        self.X['batch'], self.X['channel'], self.X['hight'], self.X['width'] = X.shape
        self.W['channel'] = self.X['channel']
        self.W['shape'] = (self.W['patch'], self.W['channel'], self.W['hight'], self.W['width'])
        #初めてXが渡されたときにのみ重みを初期化する
        if self.W['weight'] is None:
            init = FilterInitializer(self.W['shape'])
            self.W['weight'] = init.normal()
        if self.B['bias'] is None:
            self.B['bias'] = np.ones(self.B['shape'])

    def backward(self, dY):
        pass
    
    def get_output_shape(self):
        pass

class Affine(Layer2D):
    '''Affaine Layer (compatible with tensor)
    ## Arguments
    layer_size: Integer, the number of nodes to use
    activation: Activation functions to use

    ## Input shape
        4D tensor with shape:
            (batch_size, channels, hight, width)\n
        or\n
        2D tensor with shape:
            (batch_size, nodes)

    ## Output shape
        2D tensor with shape:
            (batch_size, nodes)
    '''
    def __init__(self, layer_size, activation=ReLU()):
        super().__init__(layer_size, activation)
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        super().forward(X)
        self.X['output'] = X
        return self.act.forward(np.dot(self.X['output'], self.W['weight']) + self.B['bias'])

    def backward(self, dY):
        dY = self.act.backward(dY)
        self.W['delta'] = np.dot(self.X['output'].T, dY)
        self.B['delta'] = np.sum(dY, axis=0)
        return np.dot(dY, self.W['weight'].T).reshape(self.X_shape)

class Convolution(Layer3D):
    '''Convolution Layer
    ## Arguments
    patch_size: Integer, the number of filters to use
    kernel_size: Tuple of two intengers, determins the size of the filter (hight, width)
    strides: Tuple of two intengers, (vertical stride, horizontal stride)
    padding: One of 'null', 'same'
        'null': no padding\n
        'adj' : adjust padding so that all of the input data will be convoluted\n
        'same': zero-padding the input such that the output has the same length as the input\n
        'half': zero-padding the input such that the output has the half length of the input
    activation: Activation functions to use

    ## Input shape
        4D tensor with shape:
        (batch_size, channels, hight, width)

    ## Output shape
        4D tensor with shape:
        (batch_size, patch_size, nwe_hight, new_width)
    '''
    def __init__(self, patch_size=None, kernel_size=(None,None), strides=(1,1), activation=ReLU(), padding='null', **kwargs):
        super().__init__(patch_size, kernel_size, activation)
        self.Y = {'hight':None, 'width':None}
        self.pad = {'hight':None, 'width':None}
        self.padding_option = padding
        self.strides = strides
        self.x = None
    
    def forward(self, X):
        super().forward(X)
        if self.padding_option == 'same':
            self.pad['hight'] = ((self.strides[0]-1)*self.X['hight']-self.strides[0]+self.W['hight'])
            self.pad['width'] = ((self.strides[1]-1)*self.X['width']-self.strides[1]+self.W['width'])
        elif self.padding_option == 'half':
            self.pad['hight'] = ((self.strides[0]-2)*self.X['hight']//2-self.strides[0]+self.W['hight'])
            self.pad['width'] = ((self.strides[1]-2)*self.X['width']//2-self.strides[1]+self.W['width'])
        elif self.padding_option == 'adj':
            self.pad['hight'] = (self.X['hight']-self.W['hight'])%self.strides[0]
            self.pad['width'] = (self.X['width']-self.W['width'])%self.strides[1]
        elif self.padding_option == 'null':
            self.pad['hight'] = 0
            self.pad['width'] = 0
        self.X['output'] = np.pad(self.X['input'], [(0,0), (0,0), (self.pad['hight']//2, self.pad['hight']-self.pad['hight']//2), (self.pad['width']//2, self.pad['width']-self.pad['width']//2)], 'constant', constant_values=0)

        self.Y['hight'] = (self.X['hight'] - self.W['hight'] + self.pad['hight'])//self.strides[0] + 1    
        self.Y['width'] = (self.X['width'] - self.W['width'] + self.pad['width'])//self.strides[1] + 1

        self.x = np.zeros((self.X['batch'], self.Y['hight']*self.Y['width'], self.X['channel'], self.W['hight'], self.W['width']))
        for i in range(self.Y['hight']):
            for j in range(self.Y['width']):
                self.x[:,self.Y['width']*i + j,:,:,:] = self.X['output'][:,:,i*self.strides[0]:i*self.strides[0] + self.W['hight'],j*self.strides[1]:j*self.strides[1] + self.W['width']]
        self.x = self.x.reshape(self.X['batch'], self.Y['hight'], self.Y['width'], self.X['channel'], self.W['hight'], self.W['width'])
        return self.act.forward(np.tensordot(self.x, self.W['weight'].transpose(1,2,3,0), axes=3).transpose(0,3,1,2) + self.B['bias'])

    def backward(self, dY):
        dY = self.act.backward(dY)
        self.B['delta'] = np.sum(dY, axis=0)
        self.W['delta'] = np.tensordot(dY.transpose(1,0,2,3), self.x.reshape(self.X['batch'], self.Y['hight'], self.Y['width'], self.X['channel'], self.W['hight'], self.W['width']), axes=3)
        dx = np.tensordot(dY.transpose(0,2,3,1), self.W['weight'], axes=1).reshape(self.X['batch'], self.Y['hight']*self.Y['width'], self.X['channel'], self.W['hight'], self.W['width'])
        
        self.X['delta'] = np.zeros(self.X['output'].shape)
        for i in range(self.Y['hight']):
            for j in range(self.Y['width']):
                self.X['delta'][:,:,i*self.strides[0]:i*self.strides[0] + self.W['hight'],j*self.strides[1]:j*self.strides[1] + self.W['width']] += dx[:,self.Y['width']*i + j,:,:,:]
        self.X['delta'] = self.X['delta'][:,:,self.pad['hight']//2:self.X['hight']+self.pad['hight']//2,self.pad['width']//2:self.X['width']+self.pad['width']//2]
        return self.X['delta']

class Padding(Layer3D):
    '''Padding Layer
    ## Arguments
    pad_size: Tuple of two integers, (padding hight, padding width)
    pad_value: sets a padding value (default value is zero)

    ## Input shape
        4D tensor with shape:
        (batch_size, channels, hight, width)

    ## Output shape
        4D tensor with shape:
        (batch_size, channels, padded_hight, padded_width)
    '''
    def __init__(self, pad_size=(None,None), pad_value=0):
        self.pad = {'hight':pad_size[0], 'width':pad_size[1]}
        self.pad_val = pad_value
        self.option = option
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        return np.pad(X, [(0,0),(0,0),(self.pad['hight'], self.pad['hight']),(self.pad['width'], self.pad['width'])], 'constant', constant_values=self.pad_val)

    def backward(self, dY):
        dX = np.zeros(self.X_shape)
        dX = dY[:,:,self.pad['hight']:self.X_shape[2]-self.pad['hight'],self.pad['width']:self.X_shape[3]-self.pad['width']]
        return dX

class Pooling:
    '''Pooling Layer
    ## Arguments
    pool: Tuple of two integers, that determines the hight and width of pooling window
    strides: Tuple of two integers, (vertical stride, horizontal stride)
    option: One of 'max' and 'ave'
        'max': max pooling\n
        'ave': average pooling
    padding: One of 'null', 'same'
        'null': no padding\n
        'adj' : adjust padding so that all of the input data will be convoluted\n
        'same': zero-padding the input such that the output has the same length as the input\n
        'half': zero-padding the input such that the output has the half length of the input
    '''
    def __init__(self, pool=(None,None), strides=(None,None), option='max', padding='null', **kwargs):
        self.X = {'input':None, 'output':None, 'shape':None, 'delta':None, 'batch':None, 'channel':None, 'hight':None, 'width':None}
        self.Y = {'hight':None, 'width':None}
        self.pool = {'hight':pool[0], 'width':pool[1]}
        self.strides = strides
        self.option = option
        self.pad = {'hight':None, 'width':None}
        self.padding_option = padding

        self.x = None

    def forward(self, X):
        self.X['input'] = X
        self.X['shape'] = X.shape
        self.X['batch'], self.X['channel'], self.X['hight'], self.X['width'] = X.shape

        if self.padding_option == 'same':
            self.pad['hight'] = ((self.strides[0]-1)*self.X['hight']-self.strides[0]+self.pool['hight'])
            self.pad['width'] = ((self.strides[1]-1)*self.X['width']-self.strides[1]+self.pool['width'])
        elif self.padding_option == 'half':
            self.pad['hight'] = ((self.strides[0]-2)*self.X['hight']//2-self.strides[0]+self.pool['hight'])
            self.pad['width'] = ((self.strides[1]-2)*self.X['width']//2-self.strides[1]+self.pool['width'])
        elif self.padding_option == 'adj':
            self.pad['hight'] = (self.X['hight']-self.pool['hight'])%self.strides[0]
            self.pad['width'] = (self.X['width']-self.pool['width'])%self.strides[1]
        elif self.padding_option == 'null':
            self.pad['hight'] = 0
            self.pad['width'] = 0
        self.X['input'] = np.pad(self.X['input'], [(0,0), (0,0), (self.pad['hight']//2, self.pad['hight']-self.pad['hight']//2), (self.pad['width']//2, self.pad['width']-self.pad['width']//2)], 'constant', constant_values=0)

        self.Y['hight'] = (self.X['hight'] - self.pool['hight'] + self.pad['hight'])//self.strides[0] + 1    
        self.Y['width'] = (self.X['width'] - self.pool['width'] + self.pad['width'])//self.strides[1] + 1
        
        self.x = np.zeros((self.X['batch'], self.Y['hight']*self.Y['width'], self.X['channel'], self.pool['hight'], self.pool['width']))

        for i in range(self.Y['hight']):
            for j in range(self.Y['width']):
                self.x[:,self.Y['width']*i + j,:,:,:] = self.X['input'][:,:,i*self.strides[0]:i*self.strides[0] + self.pool['hight'],j*self.strides[1]:j*self.strides[1] + self.pool['width']]
        self.X['output'] = self.x.reshape(self.X['batch'], self.Y['hight']*self.Y['width'], self.X['channel'], self.pool['hight']*self.pool['width'])
        if self.option == 'max': # max pooloing
            self.x = np.max(self.X['output'], axis=3).transpose(0,2,1)
        elif self.option == 'ave': # average pooling
            self.x = np.average(self.X['output'], axis=3).transpose(0,2,1)
        return self.x.reshape(self.X['batch'], self.X['channel'], self.Y['hight'], self.Y['width'])
    
    def backward(self, dY):
        dY = dY.reshape(self.X['batch'], self.X['channel'],-1).transpose(0,2,1)
        dx = np.zeros((self.X['batch'], self.Y['hight']*self.Y['width'], self.X['channel'], self.pool['hight']*self.pool['width']))
        if self.option == 'max':# max pooloing
            index = np.argmax(self.X['output'], axis=3).reshape(1,-1)[0]
            dY = dY.reshape(1,-1)[0]
            dx = dx.reshape(-1, self.pool['hight']*self.pool['width'])
            for i in range(len(index)):
                dx[i,index[i]] = dY[i]
        elif self.option == 'ave':# average pooling
            dY = dY.reshape(self.X['batch'], self.Y['hight']*self.Y['width'], self.X['channel'],1)
            dx = dx + dY
        
        dx = dx.reshape(self.X['batch'], self.Y['hight']*self.Y['width'], self.X['channel'], self.pool['hight'], self.pool['width'])
        self.X['delta'] =  np.zeros(self.X['input'].shape)
        for i in range(self.Y['hight']):
            for j in range(self.Y['width']):
                self.X['delta'][:,:,i*self.strides[0]:i*self.strides[0] + self.pool['hight'],j*self.strides[1]:j*self.strides[1] + self.pool['width']] += dx[:,self.Y['width']*i + j,:,:,:]
        self.X['delta'] = self.X['delta'][:,:,self.pad['hight']//2:self.X['hight']+self.pad['hight']//2,self.pad['width']//2:self.X['width']+self.pad['width']//2]
        return self.X['delta']

class Dropout:
    """Dropout Layer
    ## Arguments 
    dropout_rate: set the dropout rate
    """
    def __init__(self, dropout_rate=0.5):
        self.rate = dropout_rate
        self.mask = None    

    def __call__(self, dropout_rate=0.5):
        self.rate = dropout_rate

    def forward(self, X):
        self.mask = np.random.rand(*X.shape) < self.rate
        return X * self.mask
    
    def predict(self, X):
        return X * self.rate

    def backward(self, dY):
        return dY * self.mask
