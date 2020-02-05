from tensorflow import keras

def build_inception(input_shape, n_feature_maps, nb_classes):
    x = keras.layers.Input(shape=(input_shape))
    out = x
    return x, out