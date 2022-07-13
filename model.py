# modified code from https://github.com/jeya-maria-jose/KiU-Net-pytorch/blob/master/arch/ae.py
from block import up_block, down_block
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Activation, Add, Conv2D

def KiUnet(classes,
           n_encoder_blocks=3,
           activation='sigmoid',
           input_shape=(None,None,1),
           upsample_rates=(4,16,64),
           encoder_filters=(32,64,128),
           use_batchnorm=True):
    """
    Parameters
    ----------
    classes : int
        Number of the classes.
    n_encoder_blocks : int, optional
        Number of blocks in encoder regardless U-Net or Ki-Net. The default is 3.
    activation : str, optional
        Activation function on the last layer. The default is 'sigmoid'.
    input_shape : tuple (HxWxC), optional
        The size of input images. The default is (None,None,1).
    upsample_rates : tuple, optional
        Upsample/Downsample rates at CRFB. The default is (4, 16, 64).
    encoder_filters : tuple, optional
        Filters that used at the encoder and decoder. The default is (32,64,128).
    use_batchnorm : bool, optional
        if true then using batch normalization, if false then not using batch normalization. The default is True.

    Returns
    -------
    keras.models.Model instance.
    """

    skip_connections = [None]*(n_encoder_blocks-1)
    x_u = x_k = input = Input(shape=input_shape, name='input layer')

    #Encoder for U-Net and Ki-Net
    for i in range(n_encoder_blocks):
        x_u = down_block(encoder_filters[i], 0, i, type_='unet') (x_u)
        x_k = up_block(encoder_filters[i], 0, i, type_='kinet') (x_k)
        # Initialize skip conections
        if i != (n_encoder_blocks-1): skip_connections[i] = [x_u, x_k]

        temp_u = x_u
        temp_k = x_k
        # Cross Residual Fusion Block
        x_u = down_block(encoder_filters[i], 0, i, down_rate=upsample_rates[i],
                         type_='CRFB_u', residual=True) (temp_k)
        x_u = Add(name='CRFB_u_stage0-{}_add'.format(i)) ([x_u, temp_u])

        x_k = up_block(encoder_filters[i], 0, i, up_rate=upsample_rates[i],
                       type_='CRFB_k', residual=True) (temp_u)
        x_k = Add(name='CRFB_k_stage0-{}_add'.format(i)) ([x_k, temp_k])

    # Decoder for U-Net and Ki-Net
    for i in reversed(range(n_encoder_blocks-1)):
        x_u = up_block(encoder_filters[i], 1, i+1, type_='unet') (x_u)
        x_k = down_block(encoder_filters[i], 1, i+1, type_='kinet') (x_k)
        temp_u = x_u
        temp_k = x_k
        # Cross Residual Fusion Block
        x_u = down_block(encoder_filters[i], 1, i+1, down_rate=upsample_rates[i],
                         type_='CRFB_u', residual=True) (temp_k)
        x_u = Add(name='CRFB_u_stage1-{}_add'.format(i+1)) ([x_u, temp_u])

        x_k = up_block(encoder_filters[i], 1, i+1, up_rate=upsample_rates[i],
                       type_='CRFB_k', residual=True) (temp_u)
        x_k = Add(name='CRFB_k_stage1-{}_add'.format(i+1)) ([x_k, temp_k])

        x_u = Add(name='unet_stage1-{}_skip'.format(i+1)) ([x_u, skip_connections[i][0]])
        x_k = Add(name='kinet_stage1-{}_skip'.format(i+1)) ([x_k, skip_connections[i][1]])

    # Final Decoder
    x_u = up_block(encoder_filters[0]/2, 1, 0, type_='unet') (x_u)
    x_k = down_block(encoder_filters[0]/2, 1, 0, type_='kinet') (x_k)
    out = Add(name='final_add') ([x_u, x_k])
    out = Conv2D(classes, kernel_size=1, padding='same', activation='relu', name='final_conv') (out)
    out = Activation(activation, name='final_{}'.format(activation)) (out)

    return Model(input, out, name='KiU-Net')
