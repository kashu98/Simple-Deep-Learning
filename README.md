# Simple Deep Learning
Python implementations of Deep Learning models and algorithms with a minimum use of external library.  
畳み込み層の入力データは（バッチ数、チャネル数、高さ、幅）からなる４階テンソルとする。  
全結合層の入力データは（バッチ数、幅）からなるベクトルとする。
## Activation Functions
The following activation functions are defined in activation.py as a class that has forward and backward methods.   
活性化関数はactivation.py内に順伝播と逆伝播のメソッドを持つクラスとして定義されている。
現在活性化関数として以下のものが定義されている。
#### ReLU (Rectified Linear Unit)

forward:  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\left\{\begin{matrix}0\;&space;(x\leq&space;0)&space;\\&space;x\;&space;(x>0)&space;\end{matrix}\right." />  
backward:  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\left\{\begin{matrix}0\;&space;(x\leq&space;0)&space;\\&space;1\;&space;(x>0)&space;\end{matrix}\right." />

#### LReLU (Leaky Rectified Linear Unit)

forward:  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\left\{\begin{matrix}0.01x\;&space;(x\leq&space;0)&space;\\&space;x\;&space;\;&space;\;&space;\;&space;\;&space;\;&space;\;(x>0)&space;\end{matrix}\right."/>  
backward:  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\left\{\begin{matrix}0.01\;&space;(x\leq&space;0)&space;\\&space;1\;&space;\;&space;\;&space;\;&space;\;&space;\;&space;\;(x>0)&space;\end{matrix}\right."/>  
#### PReLU (Parameteric Rectified Linear Unit)
forward:  
<img src="https://latex.codecogs.com/gif.latex?f(\alpha&space;,x)=\left\{\begin{matrix}\alpha&space;x\;&space;(x\leq&space;0)&space;\\&space;\;&space;\;&space;x\;\;(x>0)&space;\end{matrix}\right."/>  
backward:  
<img src="https://latex.codecogs.com/gif.latex?f(\alpha&space;,x)=\left\{\begin{matrix}\alpha&space;\;&space;(x\leq&space;0)&space;\\&space;\;&space;\;&space;1\;(x>0)&space;\end{matrix}\right."/>  
#### ELU (Exponential Linear Unit)
#### SELU (Scaled Exponential Linear Unit)
#### Sigmoid (Logistic Function)

forward:  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{1&plus;e^{-x}}" />  
backward:  
<img src="https://latex.codecogs.com/gif.latex?f'(x)=f(x)(1-f(x))" />  
#### SoftPlus 
#### Tanh
#### Arctan
#### Soft sign
## Layers
#### Convolution Layer
Input:  
X(batch number, channel, hight, width)  
Karnel(patch number, channel, hight, width)  
step size (default is 1)  
padding width (default is 0)  
padding value (default is 0)  
#### Pooling Layer
#### Affine Layer
#### Maxout Layer
#### Batch Normalization Layer
#### Dropout Layer
## Loss Function
#### MAE
#### MSE
#### RMSE
