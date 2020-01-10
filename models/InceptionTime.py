import tensorflow as tf
import tensorflow.keras as keras

class InceptionBlock(keras.layers.Layer):
    def __init__(self, bottleneck_size, activation, stride, padding, nb_filters, kernel_size):
        super(InceptionBlock, self).__init__(name='Incpetion_block')
        self.bottleneck = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding=padding, activation=activation, use_bias=False)

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        self.conv_list = []

        for i in range(len(kernel_size_s)):
            self.conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False))
        self.max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')

        self.conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)

    def call(self, x):
        out1 = self.max_pool_1(x)
        out1 = self.conv_6(out1)

        out_list = []

        for conv in self.conv_list:
            out_list.append(conv(x))

        out_list.append(out1)

        out = keras.layers.Concatenate(axis=2)(out_list)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.Activation(activation='relu')(out)

        return out


class InceptionTime(keras.Model):
    def __init__(self, nb_classes, depth, bottleneck_size=32, nb_filters=32, kernel_size=41, **kwargs):
        super(InceptionTime, self).__init__(**kwargs)
        self.nb_filters = nb_filters

        self.depth = depth
        self.kernel_size = kernel_size - 1

        self.bottleneck_size = bottleneck_size

        self.gap_layer = keras.layers.GlobalAveragePooling1D()
        self.dense = keras.layers.Dense(nb_classes, activation='softmax')

    def call(self, x):
        res = x
        for d in range(self.depth):
            x = InceptionBlock(bottleneck_size=self.bottleneck_size, activation='linear', stride=1, padding='same',
                               nb_filters=self.nb_filters, kernel_size=self.kernel_size)(x)

            if d % 3 == 2:
                x_ = keras.layers.Conv1D(filters=int(res.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(x)
                x = keras.layers.BatchNormalization()(x_)
                x = keras.layers.Add()([x, res])
                x = keras.layers.Activation('relu')(x)
                res = x
        x = self.gap_layer(x)
        x = self.dense(x)
        return x