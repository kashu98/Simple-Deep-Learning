# Simple Deep Learning
畳み込み層の入力データは（バッチ数、チャネル数、高さ、幅）からなる４階テンソルとする。

全結合層の入力データは（バッチ数、幅）からなるベクトルとする。

## Activation Functions
活性化関数はactivation.py内に順伝播と逆伝播のメソッドを持つクラスとして定義されている。
現在活性化関数として以下のものが定義されている。
##### ReLU (Rectified Linear Unit)
<p align="center">
    <img src="https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_rectified_linear.svg" width="480"\>
</p>


##### LReLU (Leaky Rectified Linear Unit)
##### PReLU (Parameteric Rectified Linear Unit)
##### ELU (Exponential Linear Unit)
##### SELU (Scaled Exponential Linear Unit)
##### Sigmoid (Logistic Function)
##### SoftPlus 
##### Tanh
##### Arctan
##### Soft sign
