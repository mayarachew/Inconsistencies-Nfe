#type: ignore

import numpy as np
import random
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


class Lstm(kt.HyperModel):
    def __init__(self, vocab_size, max_sequence_length, num_labels, metrics):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.num_labels = num_labels
        self.metrics = metrics

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.vocab_size, input_length=self.max_sequence_length, output_dim=self.max_sequence_length))
        model.add(keras.layers.LSTM(hp.Int('units', min_value=16, max_value=96, step=8),
                                    hp.Choice('activation1', ['relu', 'sigmoid']),
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)))
        model.add(keras.layers.Dropout(hp.Float('rate', min_value=0.1, max_value=0.2, step=0.1),
                                       seed=SEED))
        model.add(keras.layers.Dense(self.num_labels, 
                                     hp.Choice('activation2', ['softmax'])))

        model.compile(optimizer=Adam(hp.Float("learning_rate", 1e-4, 5e-3, sampling="log")),
                      loss='categorical_crossentropy', 
                      metrics=self.metrics)

        return model


    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [8, 16, 24, 32]),
            epochs=hp.Int('epochs', min_value=4, max_value=12, step=4),
            **kwargs,
        )
        