import tensorflow as tf
import tensorflow.keras as keras

#######################Inception Model############################
class ConvBNRelu(keras.layers.Layer):
    def __init__(self, channel, kernel_size=1, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__(name='conv_block')

        self.model = keras.models.Sequential([
            keras.layers.Conv1D(channel, kernel_size, strides=strides, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        x = self.model(x, training=training)

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

    def call(self, x):
        # branch 1
        x1 = self.conv1_1(x)

        # branch 2
        x2_1 = self.conv2_1(x)
        x2 = self.conv2_2(x2_1)

        # branch 3
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x3_1)
        x3 = self.conv3_3(x3_2)

        # branch 4
        x4 = self.pool(x)
        x4 = self.pool_conv(x4)

        # concat along axis=channel
        x = tf.concat([x1, x2, x3, x4], axis=-1)

        return x

class ReductionBlk(keras.layers.Layer):

    def __init__(self, channel, strides=2):
        super(ReductionBlk, self).__init__(name='reduction_block')

        self.channel = channel
        self.strides = strides

        self.conv1_1 = ConvBNRelu(channel, kernel_size=3, strides=strides)

        self.conv2_1 = ConvBNRelu(channel, strides=1)
        self.conv2_2 = ConvBNRelu(channel, kernel_size=3, strides=1)
        self.conv2_3 = ConvBNRelu(channel, kernel_size=3, strides=strides)

        self.pool = keras.layers.MaxPooling1D(3, strides=strides, padding='same')


    def call(self, x):
        # branch 1
        x1 = self.conv1_1(x)

        # branch 2
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x2_1)
        x2 = self.conv2_3(x2_2)

        x3 = self.pool(x)


        # concat along axis=channel
        x = tf.concat([x1, x2, x3], axis=-1)

        return x

class TSCNet(keras.Model):
    def __init__(self, input_shape, num_classes, num_layers, init_channel=16, **kwargs):
        super(TSCNet, self).__init__(**kwargs)
        self.in_channel = init_channel
        self.out_channel = init_channel
        self.num_layers = num_layers
        #self.input_shape = input_shape
        self.num_classes = num_classes
        self.init_channel = init_channel

        self.conv1 = ConvBNRelu(init_channel)
        #self.blocks = keras.models.Sequential()



        self.avg_pool = keras.layers.GlobalAveragePooling1D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, x):
        out = self.conv1(x)

        for block_id in range(self.num_layers):

            for layer_id in range(2):
                if layer_id == 0:
                    out = InceptionBlk(self.out_channel, strides=1)(out)
                else:
                    out = ReductionBlk(self.out_channel, strides=2)(out)

            # enlarger out_channels per block
            self.out_channel *= 2
        #out = self.blocks(out)

        out = self.avg_pool(out)
        out = self.fc(out)

        return out

##########################Inception Model Done##############################

#######################GRU Model############################
#######################GRU Model Done############################

#######################LSTM Model############################
#######################LSTM Model Done############################

#######################Transformer Model############################
#######################Transformer Model Done############################

