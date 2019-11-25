# Code based on keras.applications repo

from ..blocks import dense_block, conv_block, transition_block

from keras import layers, models, backend

def DenseNet(input_shape,
             blocks,
             include_top=True,
             regularization='dropout',
             dropout_prob=0.2,
             dropout_noise_shape=None,
             pooling=None,
             classes=10,
             reduction=0.5):
    assert regularization in [None, 'dropout']
    assert pooling in [None, 'avg', 'max']

    inp = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inp)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    i = 2

    for block in blocks:
        x = dense_block(x, block, name='conv{}'.format(i))

        if regularization is 'dropout':
            name = 'drop{}'.format(i)
            x = layers.Dropout(rate=dropout_prob, noise_shape=dropout_noise_shape, name=name)(x)

        if i <= len(blocks):
            x = transition_block(x, reduction, name='pool{}'.format(i))
        i = i + 1

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc{}'.format(classes))(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    model = models.Model(inp, x, name='densenet')

    return model