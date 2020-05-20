from tensorflow import keras
from tensorflow.keras import layers
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

    def __call__(self, inputs, training):
        outputs = self.bi_lstm(inputs, training=training)
        return outputs

class AttentionLayer(layers.Layer):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

class AttBiLstmModel(layers.Model):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

def build_fcnablstm():
