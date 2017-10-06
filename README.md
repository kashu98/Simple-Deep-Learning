# Simple Deep Learning
This is a Python implementation of Deep Learning models and algorithms with a minimum use of external library. Simple Deep Learning aims to learn the basic concepts of deep learning by creating a library from scratch. 
## Activation Functions
The following activation functions are defined in activation.py as class that has forward and backward methods.   
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
#### SoftSign
## Layers  
The following layers are defined in layers.py as class that has forward and backward methods (someof them have predict method)
#### Convolution Layer (3D)
This layer is compatible with minibatch and deals with a 3D tensor consists of (channel, hight, width). The input data will have a shape of (batch number, channel, hight, width).  
#### Pooling Layer
Two options, max pooling and average pooling, are avalable for this layer.
#### Affine Layer
This layer is compatible with tensor expression so that you can directly connect 3D layer and fully-conected (2D) layer.
#### Maxout Layer
This layer can only be used in fully-conected (2D) layer. 
#### Batch Normalization Layer

#### Dropout Layer
## Loss Function
#### MAE (Mean Absolute Error)
#### MSE (Mean Square Error)
#### RMSE (Root Mean Square Error)

## Reference  
Following links are used as reference:  
https://en.wikipedia.org/wiki/Activation_function  
http://www.deeplearningbook.org/contents/optimization.html
