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
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
<mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mfrac linethickness="1"><mrow><mrow><mn>1</mn></mrow></mrow><mrow><mrow><mn>1</mn><mo>+</mo><msup><mrow><mi>e</mi></mrow><mrow><mrow><mo>-</mo><mi>x</mi></mrow></mrow></msup></mrow></mrow></mfrac></mrow></math>
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
