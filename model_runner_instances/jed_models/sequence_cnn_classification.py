from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from test_harness.model_runner_subclasses.mr_keras_classification import KerasClassification

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


class KerasClassificationTwoDimensional(KerasClassification):
    def __init__(self, model, model_description, training_data=None, testing_data=None,
                 data_set_description=None, train_test_split_description=None, col_to_predict='stabilityscore_2classes',
                 feature_cols_to_use=None, id_col='name', topology_col='topology', predict_untested=False, epochs=25,
                 batch_size=1000, verbose=0):
        super(KerasClassificationTwoDimensional, self).__init__(model, model_description, training_data, testing_data,
                                                                data_set_description, train_test_split_description,
                                                                col_to_predict, feature_cols_to_use, id_col,
                                                                topology_col, predict_untested)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        # print(training_data[col_to_predict].value_counts())
        value_counts = training_data[col_to_predict].value_counts().to_dict()
        num_false = value_counts[False]
        num_true = value_counts[True]
        class_weights_dict = {False: num_true, True: num_false}
        print("class_weights_dict = {}".format(class_weights_dict))
        self.class_weight = class_weights_dict

    def _fit(self, X, y):
        checkpoint_filepath = 'sequence_only_cnn_classification_{}.best.hdf5'.format(
            str(randint(1000000000, 9999999999)))
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        callbacks_list = [checkpoint_callback, stopping_callback]
        self.model.fit(np.expand_dims(np.stack([x[0] for x in X.values]), 3), y, validation_split=0.1,
                       epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks_list,
                       class_weight=self.class_weight)
        self.model.load_weights(checkpoint_filepath)
        os.remove(checkpoint_filepath)

    def _predict(self, X):
        return self.model.predict(np.expand_dims(np.stack([x[0] for x in X.values]), 3))

    def _predict_proba(self, X):
        return self._predict(X)
    
    
def sequence_only_cnn_classification(training_data, testing_data, col_to_predict, data_set_description="",
                                     train_test_split_description=""):
    amino_dict = dict(zip(
        ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
         'X', 'J', 'O'], range(23)))  # 'X' means nothing, 'J' means beginning, 'O' means end

    # training_data = pd.read_csv(training_data_file, sep=',', comment='#')
    # testing_data = pd.read_csv(testing_data_file, sep=',', comment='#')
    # untested_data = pd.read_csv(untested_data_file, sep=',', comment='#')
    training_data = training_data.copy()
    testing_data = testing_data.copy()

    MAX_RESIDUES = max(training_data.sequence.map(len).max(), testing_data.sequence.map(len).max())
    PADDING = 14  # derived from model architecture

    def make_code(sequence):
        sequence = 'X' * PADDING + 'J' + (sequence + 'O').ljust(MAX_RESIDUES + 1 + PADDING, 'X')
        code = np.zeros((23, len(sequence)))
        for i in range(len(sequence)):
            code[amino_dict[sequence[i]], i] = 1.0
        return code

    training_data['encoded_sequence'] = training_data.sequence.apply(make_code)
    training_data = training_data.sample(
        frac=1)  # shuffle data, because validation data are selected from end before shuffling
    testing_data['encoded_sequence'] = testing_data.sequence.apply(make_code)
    # untested_data['encoded_sequence'] = untested_data.sequence.apply(make_code)

    inputs = Input(shape=(23, MAX_RESIDUES + 2 + 2 * PADDING, 1))  # 22 amino acids plus null/beginning/end
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
    model = Dense(1, activation='sigmoid', kernel_regularizer=l2(.0))(model)
    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer='adam', loss='mse')

    mr = KerasClassificationTwoDimensional(model=model,
                                           model_description='Sequence CNN classification 400x5->200x9->100x17->80->40->1',
                                           col_to_predict=col_to_predict,
                                           feature_cols_to_use=['encoded_sequence'],
                                           training_data=training_data,
                                           testing_data=testing_data,
                                           # predict_untested=untested_data,
                                           batch_size=128,
                                           epochs=25,
                                           data_set_description=data_set_description,
                                           train_test_split_description=train_test_split_description)
    return mr
