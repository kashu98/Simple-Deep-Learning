# Simple Deep Learning
畳み込み層の入力データは（バッチ数、チャネル数、高さ、幅）からなる４階テンソルとする。

全結合層の入力データは（バッチ数、幅）からなるベクトルとする。

## Activation Functions
活性化関数はactivation.py内に順伝播と逆伝播のメソッドを持つクラスとして定義されている。
現在活性化関数として以下のものが定義されている。
##### ReLU (Rectified Linear Unit)
<td><a href="/wiki/File:Activation_rectified_linear.svg" class="image"><img alt="Activation rectified linear.svg" src="//upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/120px-Activation_rectified_linear.svg.png" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/180px-Activation_rectified_linear.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/240px-Activation_rectified_linear.svg.png 2x" data-file-width="120" data-file-height="60" height="60" width="120"></a></td>

##### LReLU (Leaky Rectified Linear Unit)
##### PReLU (Parameteric Rectified Linear Unit)
##### ELU (Exponential Linear Unit)
##### SELU (Scaled Exponential Linear Unit)
##### Sigmoid (Logistic Function)
##### SoftPlus 
##### Tanh
##### Arctan
##### Soft sign
