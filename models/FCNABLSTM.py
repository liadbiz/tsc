import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization

class BiLstmLayer(layers.Layer):
    def __init__(self, num_units, dropout_rate):
        super(BiLstmLayer, self).__init__()
        fwd_lstm = LSTM(num_units, return_sequences=True, name='fwd_lstm', go_backwards=False, dropout=dropout_rate)
        bwd_lstm = LSTM(num_units, return_sequences=True, name='bwd_lstm', go_backwards=True, dropout=dropout_rate)
        self.bi_lstm = Bidirectional(layer=fwd_lstm, merge_mode='sum', backward_layer=bwd_lstm)

    def call(self, inputs, training):
        outputs = self.bi_lstm(inputs, training=training)
        return outputs

class AttentionLayer(layers.Layer):
    def __init__(self, dropout_rate):
        super(AttentionLayer, self).__init__()
        self.dropout = Dropout(dropout_rate, name='attention_layer_dropout')
        #self.attention_size = attention_size

    def build(self, input_shape):
        self.attention_w = self.add_weight(name='att_w', shape=(input_shape[-1], ), initializer=tf.random_uniform_initializer(), trainable=True)
        super(AttentionLayer, self).build()


    def call(self, inputs, training):
        m = tf.tanh(inputs)
        a = tf.nn.softmax(tf.tensordot(tf.transpose(self.attention_w), m, axes=1))
        r = tf.tensordot(inputs, tf.transpose(a))
        outputs = tf.tanh(r)
        outputs = self.dropout(outputs, training=training)
        return outputs

class AttBiLstmModel(Model):
    def __init__(self, num_cells, dropout_rate):
        self.bilstm_layer = BiLstmLayer(num_cells, dropout_rate)
        self.atten_layer = AttentionLayer(dropout_rate)

    def call(self, inputs, training):
        x = self.bilstm_layer(inputs)
        out = self.atten_layer(x)
        return out


def build_fcnablstm(input_shape, num_classes, num_cells, dropout_rate):
    ip = keras.layers.Input(shape=input_shape)

    x = keras.layers.Permute((2, 1))(ip)
    x = AttBiLstmModel(num_cells, dropout_rate=0.3)(x)
    #x = keras.layers.Dropout(0.8)(x)

    y = keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)

    y = keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)

    y = keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)

    y = keras.layers.GlobalAveragePooling1D()(y)
    print(x.shape)
    x = keras.layers.concatenate([x, y])
    print(x.shape, y.shape)
    out = keras.layers.Dense(num_classes, activation='softmax')(y)
    return ip, out
