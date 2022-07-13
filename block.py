# modified code from https://github.com/jeya-maria-jose/KiU-Net-pytorch/blob/master/arch/ae.py
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D

def handle_block_names(stage, cols, type_='decoder'):
    conv_name = '{}_stage{}-{}_conv'.format(type_, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_, stage, cols)
    relu_name = '{}_stage{}-{}_relu'.format(type_, stage, cols)
    up_name = '{}_stage{}-{}_up'.format(type_, stage, cols)
    down_name = '{}_stage{}-{}_down'.format(type_, stage, cols)
    return conv_name, bn_name, relu_name, up_name, down_name

def up_block(filters, stage, cols, type_, kernel_size=(3,3),
             up_rate=2, use_batchnorm=True, residual=True):

    def layer(x):
        conv_name, bn_name, relu_name, up_name,_ = handle_block_names(stage, cols, type_=type_)

        for i in range(2):
            num_ = '_'+str(i+1)
            x = Conv2D(filters, kernel_size=kernel_size, padding='same', name=conv_name+num_) (x)
            if use_batchnorm: x = BatchNormalization(name=bn_name+num_) (x)
            x = Activation('relu', name=relu_name+num_) (x)

        if residual is True:
             x = UpSampling2D(size=up_rate, interpolation='bilinear', name=up_name) (x)

        # x = Conv2D(filters, kernel_size=kernel_size, padding='same', name=conv_name) (x)
        # if residual is False:
        #     x = UpSampling2D(size=up_rate, interpolation='bilinear', name=up_name) (x)

        # if use_batchnorm: x = BatchNormalization(name=bn_name) (x)
        # x = Activation('relu', name=relu_name) (x)
        # if residual is True:
        #      x = UpSampling2D(size=up_rate, interpolation='bilinear', name=up_name) (x)

        return x
    return layer

def down_block(filters, stage, cols, type_, kernel_size=(3,3),
               down_rate=2, use_batchnorm=True, residual=True):

    def layer(x):
        conv_name, bn_name, relu_name,_, down_name = handle_block_names(stage, cols, type_=type_)

        for i in range(2):
            num_ = '_'+str(i+1)
            x = Conv2D(filters, kernel_size=kernel_size, padding='same', name=conv_name+num_) (x)
            if use_batchnorm: x = BatchNormalization(name=bn_name+num_) (x)

            x = Activation('relu', name=relu_name+num_) (x)

        if residual is True: x = MaxPool2D(pool_size=down_rate, name=down_name) (x)

        # x = Conv2D(filters, kernel_size=kernel_size, padding='same', name=conv_name) (x)
        # if residual is False: x = MaxPool2D(pool_size=down_rate, name=down_name) (x)
        # if use_batchnorm: x = BatchNormalization(name=bn_name) (x)

        # x = Activation('relu', name=relu_name) (x)

        # if residual is True: x = MaxPool2D(pool_size=down_rate, name=down_name) (x)

        return x
    return layer
