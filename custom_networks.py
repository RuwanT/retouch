def retouch_slice_net(input_shape=(512, 512, 1), nb_classes=4):
    """

    :param input_shape: the shape of the input volume. Please note that this is channel last
    :param nb_classes: 
    :return:
    """

    from keras.models import Sequential
    from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, Input, Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Activation
    from keras.models import Model

    in_img = Input(shape=input_shape)

    conv0_1 = Conv2D(64, [3, 3], strides=(1, 1), padding='same', input_shape=input_shape)(in_img)
    conv0_1 = LeakyReLU(alpha=0.3)(conv0_1)
    conv0_2 = Conv2D(64, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv0_1)
    conv0_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv0_2)
    conv0_2 = LeakyReLU(alpha=0.3)(conv0_2)

    conv1_1 = Conv2D(128, [3, 3], strides=(1, 1), padding='same')(conv0_2)
    conv1_1 = LeakyReLU(alpha=0.3)(conv1_1)
    conv1_2 = Conv2D(128, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv1_1)
    conv1_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv1_2)
    conv1_2 = LeakyReLU(alpha=0.3)(conv1_2)

    conv2_1 = Conv2D(256, [3, 3], strides=(1, 1), padding='same')(conv1_2)
    conv2_1 = LeakyReLU(alpha=0.3)(conv2_1)
    conv2_2 = Conv2D(256, [3, 3], strides=(1, 1), padding='same')(conv2_1)
    conv2_2 = LeakyReLU(alpha=0.3)(conv2_2)
    conv2_3 = Conv2D(256, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv2_2)
    conv2_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv2_3)
    conv2_3 = LeakyReLU(alpha=0.3)(conv2_3)

    conv3_1 = Conv2D(512, [3, 3], strides=(1, 1), padding='same')(conv2_3)
    conv3_1 = LeakyReLU(alpha=0.3)(conv3_1)
    conv3_2 = Conv2D(512, [3, 3], strides=(1, 1), padding='same')(conv3_1)
    conv3_2 = LeakyReLU(alpha=0.3)(conv3_2)
    conv3_3 = Conv2D(512, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv3_2)
    conv3_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv3_3)
    conv3_3 = LeakyReLU(alpha=0.3)(conv3_3)

    conv4_1 = Conv2D(4096, [1, 1], strides=(1, 1), use_bias=False, padding='same')(conv3_3)
    conv4_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv4_1)
    conv4_1 = LeakyReLU(alpha=0.3)(conv4_1)
    conv4_1 = SpatialDropout2D(0.5)(conv4_1)
    conv4_2 = Conv2D(4096, [1, 1], strides=(1, 1), padding='same')(conv4_1)
    conv4_2 = SpatialDropout2D(0.5)(conv4_2)

    apool = GlobalAveragePooling2D(conv4_2)

    sm_IRF = Dense(2, activation='softmax')(apool)
    sm_SRF = Dense(2, activation='softmax')(apool)
    sm_PED = Dense(2, activation='softmax')(apool)

    model = Model(inputs=in_img, outputs=[sm_IRF, sm_SRF, sm_PED])

    # print model.summary()

    return model


def retouch_dual_net(input_shape=(512, 512, 1)):
    """

    :param input_shape: the shape of the input volume. Please note that this is channel last
    :param nb_classes: 
    :return:
    """

    from keras.models import Sequential
    from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, Input, Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Activation
    from keras.models import Model

    in_img = Input(shape=input_shape)

    conv0_1 = Conv2D(64, [3, 3], strides=(1, 1), padding='same', input_shape=input_shape)(in_img)
    conv0_1 = LeakyReLU(alpha=0.3)(conv0_1)
    conv0_2 = Conv2D(64, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv0_1)
    conv0_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv0_2)
    conv0_2 = LeakyReLU(alpha=0.3)(conv0_2)

    conv1_1 = Conv2D(128, [3, 3], strides=(1, 1), padding='same')(conv0_2)
    conv1_1 = LeakyReLU(alpha=0.3)(conv1_1)
    conv1_2 = Conv2D(128, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv1_1)
    conv1_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv1_2)
    conv1_2 = LeakyReLU(alpha=0.3)(conv1_2)

    conv2_1 = Conv2D(256, [3, 3], strides=(1, 1), padding='same')(conv1_2)
    conv2_1 = LeakyReLU(alpha=0.3)(conv2_1)
    conv2_2 = Conv2D(256, [3, 3], strides=(1, 1), padding='same')(conv2_1)
    conv2_2 = LeakyReLU(alpha=0.3)(conv2_2)
    conv2_3 = Conv2D(256, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv2_2)
    conv2_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv2_3)
    conv2_3 = LeakyReLU(alpha=0.3)(conv2_3)

    conv3_1 = Conv2D(512, [3, 3], strides=(1, 1), padding='same')(conv2_3)
    conv3_1 = LeakyReLU(alpha=0.3)(conv3_1)
    conv3_2 = Conv2D(512, [3, 3], strides=(1, 1), padding='same')(conv3_1)
    conv3_2 = LeakyReLU(alpha=0.3)(conv3_2)
    conv3_3 = Conv2D(512, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv3_2)
    conv3_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv3_3)
    conv3_3 = LeakyReLU(alpha=0.3)(conv3_3)

    conv4_1 = Conv2D(4096, [1, 1], strides=(1, 1), use_bias=False, padding='same')(conv3_3)
    conv4_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv4_1)
    conv4_1 = LeakyReLU(alpha=0.3)(conv4_1)
    conv4_1 = SpatialDropout2D(0.5)(conv4_1)
    conv4_2 = Conv2D(4096, [1, 1], strides=(1, 1), padding='same')(conv4_1)
    conv4_2 = SpatialDropout2D(0.5)(conv4_2)

    apool = GlobalAveragePooling2D(conv4_2)

    sm_IRF = Dense(2, activation='softmax')(apool)
    sm_SRF = Dense(2, activation='softmax')(apool)
    sm_PED = Dense(2, activation='softmax')(apool)

    model = Model(inputs=in_img, outputs=[sm_IRF, sm_SRF, sm_PED])

    # print model.summary()

    return model