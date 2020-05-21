import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization

class EmbeddingLayer(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        super(EmbeddingLayer, self).build(input_shape)
        self.W = self.add_weight(name='W',
                                 shape=(self.output_dim, self.output_dim),
                                 initializer='uniform', trainable=True)
        self.B = self.add_weight(name='B',
                                 shape=(self.input_dim, self.output_dim),
                                 initializer='uniform', trainable=True)
        self.w = self.add_weight(name='w',
                                 shape=(1, 1),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(self.input_dim, 1),
                                 initializer='uniform', trainable=True)

    def call(self, x):
        k_0 = self.w * x + self.b
        x = K.repeat_elements(x, self.output_dim, -1)
        k_i = K.sin(K.dot(x, self.W) + self.B)
        return K.concatenate([k_i, k_0], -1)


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
        self.attention_w = self.add_weight(name='att_w', shape=(input_shape[-2], ), initializer=tf.random_uniform_initializer(), trainable=True)
        super(AttentionLayer, self).build(input_shape)


    def call(self, inputs, training):
        m = tf.tanh(inputs)
        a = tf.nn.softmax(tf.matmul(tf.transpose(self.attention_w), m))
        r = tf.matmul(inputs, tf.transpose(a, perm=[0, 2, 1]))
        outputs = tf.tanh(r)
        outputs = tf.squeeze(outputs, axis=[-1])
        outputs = self.dropout(outputs, training=training)
        return outputs

class AttBiLstmModel(Model):
    def __init__(self, num_cells, dropout_rate, ts_len, embedding_size):
        super(AttBiLstmModel, self).__init__()
        self.bilstm_layer = BiLstmLayer(num_cells, dropout_rate)
        self.atten_layer = AttentionLayer(dropout_rate)
        self.embedding_layer = EmbeddingLayer(ts_len, embedding_size)

    def call(self, inputs, training):
        x = self.embedding_layer(inputs)
        x = self.bilstm_layer(x, training=training)
        x = keras.layers.Permute((2,1))(x)
        out = self.atten_layer(x, training=training)
        return out


def build_fcnablstm(input_shape, num_classes, num_cells=8, dropout_rate=0.3, embedding_size=64):
    ip = keras.layers.Input(shape=input_shape)

    #x = keras.layers.Permute((2, 1))(ip)
    x = AttBiLstmModel(num_cells=num_cells, dropout_rate=dropout_rate, ts_len=input_shape[-2], embedding_size=embedding_size)(ip)
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
    out = keras.layers.Dense(num_classes, activation='softmax')(x)
    return ip, out
