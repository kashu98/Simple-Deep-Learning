# Simple Deep Learning
畳み込み層の入力データは（バッチ数、チャネル数、高さ、幅）からなる４階テンソルとする。  
全結合層の入力データは（バッチ数、幅）からなるベクトルとする。
## Activation Functions
活性化関数はactivation.py内に順伝播と逆伝播のメソッドを持つクラスとして定義されている。
現在活性化関数として以下のものが定義されている。
##### ReLU (Rectified Linear Unit)
##### LReLU (Leaky Rectified Linear Unit)
##### PReLU (Parameteric Rectified Linear Unit)
##### ELU (Exponential Linear Unit)
##### SELU (Scaled Exponential Linear Unit)
##### Sigmoid (Logistic Function)

forward:  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{1&plus;e^{-x}}" />  
backward:  
<img src="https://latex.codecogs.com/gif.latex?f'(x)=f(x)(1-f(x))" />  
##### SoftPlus 
##### Tanh
##### Arctan
##### Soft sign
## Layers
##### Convolution Layer
Input:  
X(batch number, channel, hight, width)  
Karnel(patch number, channel, hight, width)  
step size (default is 1)  
padding width (default is 0)  
padding value (default is 0)  
##### Pooling Layer
##### Affine Layer
##### Maxout Layer
##### Batch Normalization Layer
##### Dropout Layer
## Loss Function
