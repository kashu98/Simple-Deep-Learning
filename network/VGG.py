from network import Sequential

import sys,os
sys.path.append(os.pardir)
from activation import *
from layers import *

def VGG16():
    """VGG with 16 layers
    """
    model = Sequential()
    model.add(Padding((1,1), input_shape=(3,224,224)))
    model.add(Convolutional(64, (3, 3), activation=ReLU()))
    model.add(Padding((1,1)))
    model.add(Convolutional(64, (3, 3), activation=ReLU()))
    model.add(MaxPooling((2,2), strides=(2,2)))

    model.add(Convolutional(128, 3, 3, activation=ReLU()))
    model.add(Convolutional(128, 3, 3, activation=ReLU()))
    model.add(MaxPooling((2,2), strides=(2,2)))

    model.add(Convolutional(256, 3, 3, activation=ReLU()))
    model.add(Convolutional(256, 3, 3, activation=ReLU()))
    model.add(Convolutional(256, 3, 3, activation=ReLU()))
    model.add(MaxPooling((2,2), strides=(2,2)))

    model.add(Convolutional(512, 3, 3, activation=ReLU()))
    model.add(Convolutional(512, 3, 3, activation=ReLU()))
    model.add(Convolutional(512, 3, 3, activation=ReLU()))
    model.add(MaxPooling((2,2), strides=(2,2)))

    model.add(Convolutional(512, 3, 3, activation=ReLU()))
    model.add(Convolutional(512, 3, 3, activation=ReLU()))
    model.add(Convolutional(512, 3, 3, activation=ReLU()))
    model.add(MaxPooling((2,2), strides=(2,2)))

    model.add(Affine(4096, activation=ReLU()))
    model.add(Dropout(0.5))
    model.add(Affine(4096, activation=ReLU()))
    model.add(Dropout(0.5))
    model.add(Affine(1000, activation=Softmax()))
    return model
