from tensorflow import keras

def build_fcn(input_shape, num_classes):
    x = keras.layers.Input(input_shape)
    #    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv1D(64, 3, 1, padding='same')(x)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv1D(128, 3, 1, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv1D(256, 3, 1, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    conv4 = keras.layers.Conv1D(512, 3, 1, padding='same')(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation('relu')(conv4)

    full = keras.layers.GlobalAveragePooling1D()(conv4)
    out = keras.layers.Dense(num_classes, activation='softmax')(full)

    return x, out