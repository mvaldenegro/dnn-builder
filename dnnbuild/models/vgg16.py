# Code based on keras.applications repo

from keras import layers, models, backend

# Wrappers to switch to DropConnect layers
def conv2d_wrapper(filters, kernel_size, prob=0.0, regularization='bn', **kwargs):
    if regularization is 'dropconnect':
        from keras_uncertainty.layers import DropConnectConv2D

        return DropConnectConv2D(filters, kernel_size, prob=prob, **kwargs)

    return layers.Conv2D(filters, kernel_size, **kwargs)

def dense_wrapper(units, prob=0.0, regularization='bn', **kwargs):
    if regularization is 'dropconnect':
        from keras_uncertainty.layers import DropConnectDense

        return DropConnectDense(units, prob=prob, **kwargs)
    else:
        return layers.Dense(units, **kwargs)

def VGG16(input_shape,
          include_top=True,
          regularization='bn',
          dropout_prob=0.2,
          dropout_noise_shape=None,
          neurons_fc1=4096,
          neurons_fc2=4096,
          pooling=None,
          classes=10):
    assert regularization in [None, 'dropout', 'bn', 'dropconnect']
    assert pooling in [None, 'avg', 'max']

    inp = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # Block 1
    x = conv2d_wrapper(64, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block1_conv1',
                       prob=dropout_prob,
                       regularization=regularization)(inp)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(64, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block1_conv2',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv2d_wrapper(128, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block2_conv1',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(128, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block2_conv2',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv2d_wrapper(256, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block3_conv1',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(256, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block3_conv2',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(256, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block3_conv3',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv2d_wrapper(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv1',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv2',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv3',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv2d_wrapper(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv1',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv2',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = conv2d_wrapper(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv3',
                       prob=dropout_prob,
                       regularization=regularization)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = dense_wrapper(neurons_fc1, activation='relu', name='fc1', prob=dropout_prob, regularization=regularization)(x)

        if regularization is 'bn':
            x = layers.BatchNormalization()(x)
        elif regularization is 'dropout':
            x = layers.Dropout(rate=dropout_prob, noise_shape=dropout_noise_shape)(x)

        x = dense_wrapper(neurons_fc2, activation='relu', name='fc2', prob=dropout_prob, regularization=regularization)(x)

        if regularization is 'bn':
            x = layers.BatchNormalization()(x)
        elif regularization is 'dropout':
            x = layers.Dropout(rate=dropout_prob, noise_shape=dropout_noise_shape)(x)

        x = dense_wrapper(classes, activation='softmax', name='predictions', prob=dropout_prob, regularization=regularization)(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    
    model = models.Model(inp, x, name='vgg16')

    return model