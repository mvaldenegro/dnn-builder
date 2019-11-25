# Module for residual building blocks (ResNet)

from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Dropout, BatchNormalization
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import add, Activation, Input
import keras.backend as K

def ResidualBlock(input_tensor, num_filters, kernel_size=3, stride=1, activation="relu"):
    
    x = Conv2D(num_filters, (kernel_size, kernel_size), stride=stride, padding='same', activation=activation)(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, (kernel_size, kernel_size), stride=stride, padding='same', activation=activation)(x)
    x = BatchNormalization()(x)
    z = add([x, input_tensor])
    z = Activation(activation)(z)

    return z

def ResidualBottleneckBlock(input_tensor, bottleneck_num_filters, num_filters, kernel_size=3, stride=1, activation="relu"):
    
    x = Conv2D(bottleneck_num_filters, (1, 1), stride=stride, padding='same', activation=activation)(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, (kernel_size, kernel_size), stride=stride, padding='same', activation=activation)(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, (kernel_size, kernel_size), stride=stride, padding='same', activation=activation)(x)
    x = BatchNormalization()(x)
    z = add([x, input_tensor])
    z = Activation(activation)(z)

    return z

def ResNet20(input_shape, classes):
    inp = Input(input_shape)

    x = Conv2D(16, (3, 3), padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)

    x = ResidualBlock(x, 16)
    x = ResidualBlock(x, 32)
    x = ResidualBlock(x, 64)

    