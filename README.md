# Simple Deep Learning
畳み込み層の入力データは（バッチ数、チャネル数、高さ、幅）からなる４階テンソルとする。

全結合層の入力データは（バッチ数、幅）からなるベクトルとする。

活性化関数はactivation.py内に順伝播と逆伝播のメソッドを持つクラスとして定義されている。
