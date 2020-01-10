import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.regularizers as regularizers

#######################TSC Model############################


class ConvBNRelu(keras.layers.Layer):
    def __init__(self, channel, kernel_size=1, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__(name='conv_block')

        self.model = keras.models.Sequential([
            keras.layers.Conv1D(channel, kernel_size, strides=strides, padding=padding, kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        x = self.model(x, training)

        return x


class InceptionBlk(keras.layers.Layer):
    def __init__(self, channel, strides=1):
        super(InceptionBlk, self).__init__(name='inception_block')

        self.channel = channel
        self.strides = strides

        self.conv1_1 = ConvBNRelu(channel, strides=1)

        self.conv2_1 = ConvBNRelu(channel, strides=1)

        self.conv2_2 = ConvBNRelu(channel, kernel_size=3, strides=strides)

        self.conv3_1 = ConvBNRelu(channel, strides=1)
        self.conv3_2 = ConvBNRelu(channel, kernel_size=3, strides=strides)
        self.conv3_3 = ConvBNRelu(channel, kernel_size=3, strides=strides)

        self.pool = keras.layers.MaxPooling1D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(channel, strides=1)

    def call(self, x, training=None):
        # branch 1
        x1 = self.conv1_1(x, training=training)

        # branch 2
        x2_1 = self.conv2_1(x, training=training)
        x2 = self.conv2_2(x2_1, training=training)

        # branch 3
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
        x3 = self.conv3_3(x3_2, training=training)

        # branch 4
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)

        # concat along axis=channel
        x = tf.concat([x1, x2, x3, x4], axis=-1)

        return x


class ReductionBlk(keras.layers.Layer):
    def __init__(self, channel, strides=2):
        super(ReductionBlk, self).__init__(name='reduction_block')

        self.channel = channel
        self.strides = strides

        self.conv1_1 = ConvBNRelu(channel, kernel_size=3, strides=strides, padding='valid')

        self.conv2_1 = ConvBNRelu(channel, strides=1)
        self.conv2_2 = ConvBNRelu(channel, kernel_size=3, strides=1)
        self.conv2_3 = ConvBNRelu(channel, kernel_size=3, strides=strides, padding='valid')

        self.pool = keras.layers.MaxPooling1D(3, strides=strides, padding='valid')

    def call(self, x, training=None):
        # branch 1
        x1 = self.conv1_1(x, training=training)

        # branch 2
        x2_1 = self.conv2_1(x, training=training)
        x2_2 = self.conv2_2(x2_1, training=training)
        x2 = self.conv2_3(x2_2, training=training)

        x3 = self.pool(x)


        # concat along axis=channel
        x = tf.concat([x1, x2, x3], axis=-1)

        return x


class TSCNet(keras.Model):
    def __init__(self, num_classes, num_layers, bottleneck_channel=16, **kwargs):
        super(TSCNet, self).__init__(**kwargs)
        self.bottleneck_channel = bottleneck_channel
        self.num_layers = num_layers
        #self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = ConvBNRelu(bottleneck_channel, kernel_size=3, strides=1, padding='same')
        self.conv2 = ConvBNRelu(bottleneck_channel, kernel_size=3, strides=2, padding='valid')
        self.conv3 = ConvBNRelu(bottleneck_channel, kernel_size=3, strides=1, padding='same')
        self.conv4 = ConvBNRelu(bottleneck_channel, kernel_size=3, strides=2, padding='valid')

        self.blocks = keras.models.Sequential()

        #assert self.num_layers % 3 == 2
        for block_id in range(1, self.num_layers + 1):

            if block_id % 3 == 0:
                block = ReductionBlk(self.bottleneck_channel, strides=2)
            else:
                block = InceptionBlk(self.bottleneck_channel, strides=1)
            self.blocks.add(block)
            # enlarger out_channels per block
            # self.out_channel *= 2

        self.avg_pool = keras.layers.GlobalAveragePooling1D()
        self.fc = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        out = self.conv3(out, training=training)
        out = self.conv4(out, training=training)

        out = self.blocks(out, training=training)

        out = self.avg_pool(out)
        out = self.fc(out)

        return out

##########################TSC Model Done##############################


#######################InceptionTIme Model############################

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
#######################InceptionTime Model Done############################





#######################GRU Model############################
#######################GRU Model Done############################

#######################LSTM Model############################
#######################LSTM Model Done############################

#######################Transformer Model############################
#######################Transformer Model Done############################

