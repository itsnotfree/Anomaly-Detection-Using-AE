import tensorflow as tf
from tensorflow import keras


class AEfunctions(object):
    def train(self, x_train, x_test, epochs, batch_size):
        self.autoencoder.fit(x_train, x_train,
                             epochs,
                             batch_size,
                             shuffle=True,
                             validation_data=(x_test, x_test))
    def encode(self, x):
        return self.encoder.predict(x)
    def decode(self, x):
        return self.decoder.predict(x)
    def summary(self):
        self.autoencoder.summary()
    def prediction(self,x):
        encoded=self.encoder.predict(x)
        return self.decoder.predict(encoded)