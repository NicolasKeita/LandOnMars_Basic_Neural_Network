# import random
# import logging
# import pandas as pd
import numpy as np  # TODO FIX
# from sklearn.metrics import accuracy_score
# import keras_core as keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras_core import optimizers


def create_neural_network() -> Sequential:
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
