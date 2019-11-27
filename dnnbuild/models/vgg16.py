# Code based on keras.applications repo

from keras import layers, models, backend

def VGG16(input_shape,
          include_top=True,
          regularization='bn',
          dropout_prob=0.2,
          dropout_noise_shape=None,
          neurons_fc1=4096,
          neurons_fc2=4096,
          pooling=None,
          classes=10):
    assert regularization in [None, 'dropout', 'bn']
    assert pooling in [None, 'avg', 'max']

    inp = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inp)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)

    if regularization is 'bn':
        x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(neurons_fc1, activation='relu', name='fc1')(x)

        if regularization is 'bn':
            x = layers.BatchNormalization()(x)
        elif regularization is 'dropout':
            x = layers.Dropout(rate=dropout_prob, noise_shape=dropout_noise_shape)(x)

        x = layers.Dense(neurons_fc2, activation='relu', name='fc2')(x)

        if regularization is 'bn':
            x = layers.BatchNormalization()(x)
        elif regularization is 'dropout':
            x = layers.Dropout(rate=dropout_prob, noise_shape=dropout_noise_shape)(x)

        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    
    model = models.Model(inp, x, name='vgg16')

    return model