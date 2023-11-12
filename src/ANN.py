# import random
# import logging
# import pandas as pd
import numpy as np  # TODO FIX
# from sklearn.metrics import accuracy_score
# import keras_core as keras

from keras_core import optimizers


def create_neural_network() -> Sequential:
    feature_amount = 7
    action_size = 2
    number_of_action = 100
    neurons_layer_1 = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons_layer_1, activation='relu', input_shape=(feature_amount,)),
        tf.keras.layers.Dense(action_size * number_of_action, activation='tanh')
    ])
    return model


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(output_dim * 100, activation='tanh')  # Output sequence of 100 actions
    ])
    return model


    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class ANN(Sequential):

    def __init__(self, child_weights=None):
        super().__init__()

        if child_weights is None:
            layer1 = Dense(8, input_shape=(8,), activation='sigmoid')
            layer2 = Dense(1, activation='sigmoid')
            self.add(layer1)
            self.add(layer2)
        else:
            self.add(
                Dense(
                    8,
                    input_shape=(8,),
                    activation='sigmoid',
                    weights=[child_weights[0], np.ones(8)])
            )
            self.add(
                Dense(
                    1,
                    activation='sigmoid',
                    weights=[child_weights[1], np.zeros(1)])
            )

    def forward_propagation(self, train_feature, train_label):
        predict_label = self.predict(train_feature.values)
        self.fitness = accuracy_score(train_label, predict_label.round())  # TODO fix warning define fitness in __init__
