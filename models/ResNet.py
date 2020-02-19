from tensorflow import keras
from keras.regularizers import l2

def build_resnet(input_shape, n_feature_maps, nb_classes):
    x = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps, 1, 1, padding='same')(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x)

    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)


    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)


    conv_y = keras.layers.Conv1D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)


    conv_z = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, 1, 1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)

    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)


    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)


    conv_y = keras.layers.Conv1D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)


    conv_z = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, 1, 1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)

    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    full = keras.layers.GlobalAveragePooling1D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print('        -- model was built.')
    return x, out


def basic_block(filters, init_strides=1, is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = keras.layers.Conv1D(filters=filters, kernel_size=7,
                           strides=init_strides,
                           padding="same",
                           )(input)
        else:
            conv1 = bn_relu_conv(nb_filter=filters, kernel_size=3,
                                  strides=init_strides)(input)
        conv2 = bn_relu_conv(nb_filter=filters, kernel_size=3, strides=init_strides)(conv1)
        residual = bn_relu_conv(nb_filter=filters, kernel_size=3, strides=init_strides)(conv2)
        return shortcut(input, residual)

    return f

def bottleneck_block(filters, init_strides=1, is_first_block_of_first_layer=False):
    """bottleneck blocks for use on resnets with layers > 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = keras.layers.Conv1D(filters=filters, kernel_size=1,
                           strides=init_strides,
                           padding="same",
                           )(input)
        else:
            conv1 = bn_relu_conv(nb_filter=filters, kernel_size=1,
                                  strides=init_strides)(input)

        conv3 = bn_relu_conv(nb_filter=filters, kernel_size=3)(conv1)
        residual = bn_relu_conv(nb_filter=filters*4, kernel_size=1, strides=init_strides)(conv3)
        return shortcut(input, residual)

    return f


def shortcut(input, residual):
    input_with = input.shape[1]
    residual_with = residual.shape[1]
    stride = int(round(input_with / residual_with))
    short = input
    if stride > 1 or input.shape[-1] != residual.shape[-1]:
        short = keras.layers.Conv1D(filters=residual.shape[-1],
                                       kernel_size=1,
                                       strides=stride,
                                       padding="valid",
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=l2(0.0001))(input)
    print(short.shape, residual.shape)
    return keras.layers.Add()([short, residual])


def bn_relu(x):
    norm = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation('relu')(norm)


def bn_relu_conv(nb_filter, kernel_size, strides=1, padding='same'):
    """
    bn-relu-filter
    see http://arxiv.org/pdf/1603.05027v2.pdf for more details
    """
    def f(x):
        acti = bn_relu(x)
        return keras.layers.Conv1D(nb_filter, kernel_size, strides, padding, kernel_initializer='he_normal',
                               kernel_regularizer=l2(0.0001))(acti)
    return f


def conv_bn_relu(nb_filter, kernel_size, strides, padding='same'):
    """
    filer-bn-relu
    see http://arxiv.org/pdf/1603.05027v2.pdf for more details
    """
    def f(x):
        fea = keras.layers.Conv1D(nb_filter, kernel_size, strides, padding, kernel_initializer='he_normal',
                              kernel_regularizer=l2(0.0001))(x)
        return bn_relu(fea)
    return f

def residual_block(filters, repetitions, is_first_layer, block_func):
    def f(input):
        for i in range(repetitions):
            init_strides = 1
            if i == 0 and not is_first_layer:
                init_strides = 2
            if block_func == 'basic':
                input = basic_block(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
            if block_func == 'bottleneck':
                input = bottleneck_block(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_classes, block_func, repetitions):
        """build custom resnet architecture like network

        :param input_shape: input shape of input data
        :param num_classes: number of class in final softmax layer
        :param block_func block function name
        :param repetitions: number of repetitions of each block
        :return: a keras model
        """
        input = keras.layers.Input(shape=(input_shape))
        print(input.shape)
        conv1 = conv_bn_relu(nb_filter=64, kernel_size=7, strides=2)(input)
        print(conv1.shape)
        pool1 = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv1)
        print(pool1.shape)
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            print(i, r)
            block = residual_block(filters=filters, repetitions=r, is_first_layer=(i == 0), block_func=block_func)(block)
            filters *= 2
        block = bn_relu(block)
        full = keras.layers.GlobalAveragePooling1D()(block)
        out = keras.layers.Dense(num_classes, activation='softmax')(full)
        print('        -- model was built.')
        return input, out

def build_resnet10(input_shape, num_classes):
    repetitions = [1, 1, 1, 1]
    return ResnetBuilder.build(input_shape, num_classes, 'basic', repetitions)

def build_resnet18(input_shape, num_classes):
    repetitions = [2, 2, 2, 2]
    return ResnetBuilder.build(input_shape, num_classes, 'basic', repetitions)

def build_resnet34(input_shape, num_classes):
    repetitions = [3, 4, 6, 3]
    return ResnetBuilder.build(input_shape, num_classes, 'basic', repetitions)

def build_resnet50(input_shape, num_classes):
    repetitions = [3, 4, 6, 3]
    return ResnetBuilder.build(input_shape, num_classes, 'bottleneck', repetitions)



