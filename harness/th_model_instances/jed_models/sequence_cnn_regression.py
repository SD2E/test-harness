from harness.th_model_classes.class_keras_regression import KerasRegression

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from random import randint
import os


class KerasRegressionTwoDimensional(KerasRegression):
    def __init__(self, model, model_author, model_description, epochs=25, batch_size=128, verbose=0):
        super(KerasRegressionTwoDimensional, self).__init__(model, model_author, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        checkpoint_filepath = 'sequence_only_cnn_{}.best.hdf5'.format(str(randint(1000000000, 9999999999)))
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        callbacks_list = [checkpoint_callback, stopping_callback]
        # Pulling out the zeroth item from each element because X is a dataframe and 
        # so each item in X.values is a list of length 1. Same for _predict, below.
        self.model.fit(np.expand_dims(np.stack([x[0] for x in X.values]), 3), y, validation_split=0.1,
                       epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks_list)
        self.model.load_weights(checkpoint_filepath)
        os.remove(checkpoint_filepath)

    def _predict(self, X):
        return self.model.predict(np.expand_dims(np.stack([x[0] for x in X.values]), 3))


def sequence_only_cnn(max_residues, padding):
    amino_inputs = Input(shape=(23, max_residues + 2 + 2 * padding, 1))  # 22 amino acids plus null/beginning/end

    amino_model = Conv2D(400, (23, 5), kernel_regularizer=l2(.0), activation='relu')(amino_inputs)
    amino_model = Dropout(0.3)(amino_model)
    amino_model = Conv2D(200, (1, 9), kernel_regularizer=l2(.0), activation='relu')(amino_model)
    amino_model = Dropout(0.3)(amino_model)
    amino_model = Conv2D(100, (1, 17), kernel_regularizer=l2(.0), activation='relu')(amino_model)
    amino_model = Dropout(0.3)(amino_model)

    model = Flatten()(amino_model)

    model = Dense(80, activation='elu', kernel_regularizer=l2(.0))(model)
    model = Dropout(0.3)(model)
    model = Dense(40, activation='elu', kernel_regularizer=l2(.0))(model)
    model = Dense(1, activation='linear', kernel_regularizer=l2(.0))(model)
    model = Model(inputs=amino_inputs, outputs=model)
    model.compile(optimizer='adam', loss='mse')

    th_model = KerasRegressionTwoDimensional(model=model, model_author="Jed",
                                             model_description='Sequence CNN regressor: 400x5->200x9->100x17->80->40->1',
                                             batch_size=128, epochs=25)
    return th_model
