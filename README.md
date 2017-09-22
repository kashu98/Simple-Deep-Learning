# Simple Deep Learning
***
畳み込み層の入力データは（バッチ数、チャネル数、高さ、幅）からなる４階テンソルとする。  
全結合層の入力データは（バッチ数、幅）からなるベクトルとする。
![Qiita](https://qiita-image-store.s3.amazonaws.com/0/45617/015bd058-7ea0-e6a5-b9cb-36a4fb38e59c.png "Qiita")
## Activation Functions
活性化関数はactivation.py内に順伝播と逆伝播のメソッドを持つクラスとして定義されている。
現在活性化関数として以下のものが定義されている。
##### ReLU (Rectified Linear Unit)
##### LReLU (Leaky Rectified Linear Unit)
##### PReLU (Parameteric Rectified Linear Unit)
##### ELU (Exponential Linear Unit)
##### SELU (Scaled Exponential Linear Unit)
##### Sigmoid (Logistic Function)
##### SoftPlus 
##### Tanh
##### Arctan
##### Soft sign
## Layers
##### Convolution Layer
##### Pooling Layer
##### Affine Layer
##### Maxout Layer
##### Batch Normalization Layer
##### Dropout Layer
## Loss Function
