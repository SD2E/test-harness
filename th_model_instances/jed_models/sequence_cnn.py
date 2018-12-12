from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from test_harness.th_model_classes.class_keras_regression import KerasRegression

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate, BatchNormalization
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from random import randint
import os
from keras import backend as K


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))


class KerasRegressionTwoDimensional(KerasRegression):
    def __init__(self, model, model_description, epochs=25, batch_size=1000, verbose=0):
        super(KerasRegressionTwoDimensional, self).__init__(model, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        checkpoint_filepath = 'sequence_only_cnn_{}.best.hdf5'.format(str(randint(1000000000, 9999999999)))
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        callbacks_list = [checkpoint_callback, stopping_callback]
        self.model.fit(np.expand_dims(np.stack([x[0] for x in X.values]), 3), y, validation_split=0.1,
                       epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks_list)
        self.model.load_weights(checkpoint_filepath)
        os.remove(checkpoint_filepath)

    def _predict(self, X):
        return self.model.predict(np.expand_dims(np.stack([x[0] for x in X.values]), 3))


def sequence_only_cnn(max_residues, padding):
    inputs = Input(shape=(23, max_residues + 2 + 2 * padding, 1))  # 22 amino acids plus null/beginning/end
    amino_inputs = Lambda(lambda x: x[:, :23, :, :])(inputs)

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
    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer='adam', loss='mse')

    mr = KerasRegressionTwoDimensional(model=model,
                                       model_description='Sequence CNN 400x5->200x9->100x17->80->40->1',
                                       batch_size=128,
                                       epochs=25)
    return mr


# TODO: make single function that calls the two data_wrangling functions, returns edited dataframe. Move the other two functions here too.