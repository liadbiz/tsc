import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import backend as K

class AttentionLSTMCell(layers.Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 attention_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 attention_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 attention_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_attention=False,
                 implementation=1,
                 **kwargs):
        super(AttentionLSTMCell, self).__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=3)]
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.attention_activation = activations.get(attention_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.attention_regularizer = regularizers.get(attention_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attention_constraint = constraints.get(attention_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.return_attention = return_attention
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.implementation = implementation
        self.state_spec = [layers.InputSpec(shape=(None, self.units)),
                           layers.InputSpec(shape=(None, self.units))]
        self.state_size = (self.units, self.units)

    def build(self, input_shape):

        if hasattr(self, 'timesteps') and self.timesteps is not None:
            self.timestep_dim = self.timesteps
        else:
            self.timestep_dim = 1  # input_shape[0]

        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # add attention kernel
        self.attention_kernel = self.add_weight(
            shape=(self.input_dim, self.units * 4),
            name='attention_kernel',
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint)

        # add attention weights
        # weights for attention model
        self.attention_weights = self.add_weight(shape=(self.input_dim, self.units),
                                                 name='attention_W',
                                                 initializer=self.attention_initializer,
                                                 regularizer=self.attention_regularizer,
                                                 constraint=self.attention_constraint)

        self.attention_recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                           name='attention_U',
                                                           initializer=self.recurrent_initializer,
                                                           regularizer=self.recurrent_regularizer,
                                                           constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.attention_bias = self.add_weight(shape=(self.units,),
                                                  name='attention_b',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)

            self.attention_recurrent_bias = self.add_weight(shape=(self.units, 1),
                                                            name='attention_v',
                                                            initializer=self.bias_initializer,
                                                            regularizer=self.bias_regularizer,
                                                            constraint=self.bias_constraint)
        else:
            self.bias = None
            self.attention_bias = None
            self.attention_recurrent_bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        self.attention_i = self.attention_kernel[:, :self.units]
        self.attention_f = self.attention_kernel[:, self.units: self.units * 2]
        self.attention_c = self.attention_kernel[:, self.units * 2: self.units * 3]
        self.attention_o = self.attention_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

        self.built = True

    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(4)]
        else:
            self._dropout_mask = None


    def _generate_recurrent_dropout_mask(self, inputs, training=None):
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._recurrent_dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(4)]
        else:
            self._recurrent_dropout_mask = None

    def call(self, inputs, states, training=None):
        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # alignment model
        h_att = K.repeat(h_tm1, self.timestep_dim)
        att = _time_distributed_dense(inputs, self.attention_weights, self.attention_bias,
                                      input_dim=self.input_dim, output_dim=self.units, timesteps=self.timestep_dim)
        attention_ = self.attention_activation(K.dot(h_att, self.attention_recurrent_weights) + att)  # energy
        attention_ = K.squeeze(K.dot(attention_, self.attention_recurrent_bias), 2)  # energy

        alpha = K.exp(attention_)

        if dp_mask is not None:
            alpha *= dp_mask[0]

        alpha /= K.sum(alpha, axis=1, keepdims=True)
        alpha_r = K.repeat(alpha, self.input_dim)
        alpha_r = K.permute_dimensions(alpha_r, (0, 2, 1))

        # make context vector (soft attention after Bahdanau et al.)
        z_hat = inputs * alpha_r
        context_sequence = z_hat
        z_hat = K.sum(z_hat, axis=1)

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i) + K.dot(z_hat, self.attention_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.recurrent_kernel_f) + K.dot(z_hat, self.attention_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernel_c) + K.dot(z_hat, self.attention_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o) + K.dot(z_hat, self.attention_o))
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            z += K.dot(z_hat, self.attention_kernel)

            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]


class AttentionLSTM(layers.RNN):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 attention_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 attention_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 attention_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 return_attention=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
            implementation = 1

        if K.backend() == 'cntk':
            if not kwargs.get('unroll') and (dropout > 0 or recurrent_dropout > 0):
                warnings.warn(
                    'RNN dropout is not supported with the CNTK backend '
                    'when using dynamic RNNs (i.e. non-unrolled). '
                    'You can either set `unroll=True`, '
                    'set `dropout` and `recurrent_dropout` to 0, '
                    'or use a different backend.')
                dropout = 0.
                recurrent_dropout = 0.

        cell = AttentionLSTMCell(units,
                                 activation=activation,
                                 recurrent_activation=recurrent_activation,
                                 attention_activation=attention_activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 recurrent_initializer=recurrent_initializer,
                                 attention_initializer=attention_initializer,
                                 bias_initializer=bias_initializer,
                                 unit_forget_bias=unit_forget_bias,
                                 kernel_regularizer=kernel_regularizer,
                                 recurrent_regularizer=recurrent_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 activity_regularizer=activity_regularizer,
                                 attention_regularizer=attention_regularizer,
                                 kernel_constraint=kernel_constraint,
                                 recurrent_constraint=recurrent_constraint,
                                 bias_constraint=bias_constraint,
                                 attention_constraint=attention_constraint,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_attention=return_attention,
                                 implementation=implementation)
        super(AttentionLSTM, self).__init__(cell,
                                            return_sequences=return_sequences,
                                            return_state=return_state,
                                            go_backwards=go_backwards,
                                            stateful=stateful,
                                            unroll=unroll,
                                            **kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        self.cell.timesteps = input_shape[1]
        self.cell.build(input_shape)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._generate_dropout_mask(inputs, training=training)
        self.cell._generate_recurrent_dropout_mask(inputs, training=training)
        return super(AttentionLSTM, self).call(inputs,
                                               mask=mask,
                                               training=training,
                                               initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def attention_activation(self):
        return self.cell.attention_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def attention_initializer(self):
        return self.cell.attention_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def activity_regularizer(self):
        return self.cell.activity_regularizer

    @property
    def attention_regularizer(self):
        return self.cell.attention_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def attention_constraint(self):
        return self.cell.attention_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'attention_activation': activations.serialize(self.attention_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'attention_initializer': initializers.serialize(self.attention_initializer),
                  'use_chrono_initialization': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'attention_regularizer': regularizers.serialize(self.attention_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'attention_constraint': constraints.serialize(self.attention_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'return_attention': self.return_attention}
        base_config = super(AttentionLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


def build_fcnalstm(input_shape, nb_class, num_cells=8):

    ip = layers.Input(shape=input_shape)

    x = layers.Permute((2,1))(ip)
    x = layers.AttentionLSTM(num_cells)(x)
    x = layers.Dropout(0.8)(x)

    y = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.GlobalAveragePooling1D()(y)

    x = layers.concatenate([x, y])

    out = layers.Dense(nb_class, activation='softmax')(x)

    return ip, out