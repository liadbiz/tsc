import tensorflow as tf
from tensorflow import keras

class ConvBNRelu(keras.layers.Layer):
    def __init__(self, channel, kernel_size=1, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__(name='conv_block')

        self.model = keras.models.Sequential([
            keras.layers.Conv1D(channel, kernel_size, strides=strides, padding=padding),
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

def build_tscnet(input_shape, num_classes):
    x = keras.layers.Input(shape=(input_shape))