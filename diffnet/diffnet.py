import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class DiffNET(keras.Model):
    def __init__(self, **kwargs):
        super().__init__()

        default_params = {
            "training" : False,
            "dropout" : 0.5,
            "input_size" : 5,
            "transfer" : False,
        }

        params = {}
        for param in default_params.keys():
            if param in kwargs:
                params[param] = kwargs[param]
            else:
                params[param] = default_params[param]

        self.dropout_rate = params["dropout"]
        self.input_size = params["input_size"]
        self.transfer = params["transfer"]

        self.dropout_layer = layers.Dropout(self.dropout_rate)
        
        self.fc1 = layers.Dense(units=512, kernel_regularizer=keras.regularizers.l2(0.01), input_shape=(self.input_size,), activation="relu")
        self.fc2 = layers.Dense(units=128, kernel_regularizer=keras.regularizers.l2(0.01), activation="relu")
        self.out = layers.Dense(units=1)
        
        self.fc1.trainable = not(self.transfer)
        
    def initialize_weights(self):
        self(np.zeros([1, self.input_size[0]], dtype=np.float32))
        self.built=True
    
    @tf.function
    def call(self, x, training=False):          
        # frozen layer
        out = self.fc1(x)

        # transfer layer
        out = self.fc2(out)
        out = self.dropout_layer(out, training=training)
        out = self.out(out)

        return out

