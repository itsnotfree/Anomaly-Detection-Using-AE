
from aeFunctions import AEfunctions
from keras.layers import Input, Dense
from keras.models import Model


class Autoencoder(AEfunctions):
    def __init__(self, dim_in, encoding_dim):
        input_vec = Input(shape=(dim_in,), name='EncoderIn')
        encoded = Dense(encoding_dim, activation='relu', name='Encoder')(input_vec)
        decoded = Dense(dim_in, activation='sigmoid', name='Decoder')(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec,encoded)
        encoded_input = Input(shape=(encoding_dim,), name='DecoderIn')
        decoder_layer = self.autoencoder.layers[-1]             #gets "decoded" layer
        self.decoder = Model(encoded_input,decoder_layer(encoded_input))
        self.autoencoder.compile(optimizer='adam', loss='mse')