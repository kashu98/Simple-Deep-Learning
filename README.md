# Simple Deep Learning
畳み込み層の入力データは（バッチ数、チャネル数、高さ、幅）からなる４階テンソルとする。

全結合層の入力データは（バッチ数、幅）からなるベクトルとする。

活性化関数はactivation.py内に順伝播と逆伝播のメソッドを持つクラスとして定義されている。
現在活性化関数として以下のものが定義されている。
ReLU
LReLU
PReLU
ELU
SELU
Sigmoid
SoftPlus
Tanh
Arctan
Soft sign
