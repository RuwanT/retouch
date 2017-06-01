

def multiclass_balanced_cross_entropy_loss(y_true, y_pred):
    import keras.backend as K

    # TODO: Check behavior when all classes are not present
    shape = list(K.shape(y_pred))
    batch_size = shape[0]
    num_classes = shape[-1]

    y_pred_ = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

    cross_ent = (K.log(y_pred_) * y_true)
    cross_ent = K.sum(cross_ent, axis=-2, keepdims=False)
    cross_ent = K.sum(cross_ent, axis=-2, keepdims=False)
    cross_ent = K.reshape(cross_ent, shape=(batch_size, num_classes))

    y_true_ = K.sum(y_true, axis=-2, keepdims=False)
    y_true_ = K.sum(y_true_, axis=-2, keepdims=False)
    y_true_ = K.reshape(y_true_, shape=(batch_size, num_classes)) + K.ones(shape=(batch_size, num_classes))

    cross_ent = cross_ent / y_true_

    return - K.mean(cross_ent, axis=-1, keepdims=False)


def retouch_dual_net(input_shape=(512, 512, 1)):
    """

    :param input_shape: the shape of the input volume. Please note that this is channel last
    :param nb_classes: 
    :return:
    """

    from keras.models import Sequential
    from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, Input, Dense, UpSampling2D
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import concatenate
    from keras.layers import Activation
    from keras.models import Model
    from keras.utils import plot_model
    from custom_layers import Softmax4D
    from keras.optimizers import SGD, Adam

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

    apool = GlobalAveragePooling2D(data_format='channels_last')(conv4_2)

    # Slice classification outputs
    sm_IRF = Dense(2, activation='softmax', name='sm_IRF')(apool)
    sm_SRF = Dense(2, activation='softmax', name='sm_SRF')(apool)
    sm_PED = Dense(2, activation='softmax', name='sm_PED')(apool)

    # Segmentation layers
    seg_1 = UpSampling2D(size=(2, 2))(conv0_2)
    seg_2 = UpSampling2D(size=(4, 4))(conv1_2)
    seg_3 = UpSampling2D(size=(8, 8))(conv2_3)
    seg_4 = UpSampling2D(size=(16, 16))(conv3_3)
    seg_5 = UpSampling2D(size=(16, 16))(conv4_1)

    concat_1 = concatenate([seg_1, seg_2, seg_3, seg_4, seg_5], axis=-1)
    seg_out = Conv2D(4, [1, 1], strides=(1, 1), padding='same', activation='relu')(concat_1)
    seg_out = Softmax4D(axis=-1, name='seg_out')(seg_out)

    model = Model(inputs=in_img, outputs=[sm_IRF, sm_SRF, sm_PED, seg_out])

    # print model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    sgd = SGD(lr=0.001, momentum=0.5, decay=1e-6, nesterov=False)
    model.compile(optimizer=sgd, loss={'sm_IRF': 'categorical_crossentropy', 'sm_SRF': 'categorical_crossentropy',
                                       'sm_PED': 'categorical_crossentropy',
                                       'seg_out': multiclass_balanced_cross_entropy_loss})

    return model
