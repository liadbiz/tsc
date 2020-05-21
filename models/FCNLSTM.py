from tensorflow import keras

def build_fcnlstm(input_shape, num_classes, num_cells):
    ip = keras.layers.Input(shape=input_shape)

    #x = keras.layers.Permute((2, 1))(ip)
    x = keras.layers.LSTM(num_cells)(ip)
    x = keras.layers.Dropout(0.8)(x)


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