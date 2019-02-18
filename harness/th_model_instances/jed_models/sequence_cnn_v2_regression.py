from harness.th_model_classes.class_keras_regression import KerasRegression

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from random import randint, randrange
import os


class KerasRegressionTwoDimensional(KerasRegression):
    def __init__(self, model, model_description, epochs=25, batch_size=128, verbose=0):
        super(KerasRegressionTwoDimensional, self).__init__(model, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        X = np.expand_dims(np.stack([x for x in X.values]), 3)
        y_stability = np.stack([x[0] for x in y.values], axis=1).T
        y_dssp = np.squeeze(np.stack([x[1] for x in y.values], axis=1))
        y = [y_stability, y_dssp]
        val_size = int(X.shape[0]*.1)
        Xv = X[-val_size:, :, :, :]
        yv = [y[0][-val_size:, :], y[1][-val_size:, :, :]]
        X = X[:-val_size, :, :, :]
        y = [y[0][:-val_size, :], y[1][:-val_size, :, :]]

        def data_gen(batch_size):
            batch_ind = 0
            while True:
                xi = randrange(X.shape[0] - batch_size)
                if batch_ind % batch_size == 0:
                    batch_ind = 0
                    x_ret = []
                    y_ret = [[], []]
                x = X[xi, :, :, :]
                y0 = y[0][xi, :]
                y1 = y[1][xi, :, :]
                minshift = np.argmax(x[amino_dict['O'], :, :]) - x.shape[1] + PADDING
                maxshift = np.argmax(x[amino_dict['J'], :, :]) - PADDING
                shift = randrange(minshift, maxshift) + 1 # +1 is because we want to be able to shift maxshift (putting the 'J' at the beginning) but not minshift (putting the 'O' wrapped around and at the beginning - we want the farthest rightward shift possible to put the 'O' at the end)
                x_ret += [np.roll(x, shift, axis=1)]
                y_ret[0] += [y0]
                y_ret[1] += [np.roll(y1, shift, axis=0)]
                batch_ind += 1
                if batch_ind % batch_size == 0:
                    yield np.stack(x_ret), [np.stack(y_ret[0]), np.stack(y_ret[1])]

        checkpoint_filepath = 'sequence_only_cnn_v2_{}.best.hdf5'.format(str(randint(1000000000, 9999999999)))
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5)
        callbacks_list = [checkpoint_callback, stopping_callback]
        self.model.fit_generator(
                                 data_gen(self.batch_size),
                                 epochs=self.epochs,
                                 steps_per_epoch=1*X.shape[0]/self.batch_size,
                                 validation_data=(Xv, yv),
                                 callbacks=callbacks_list,
                                 verbose=self.verbose
                                )

        # Based on permissible transitions between DSSP codes
        transition_kernels = K.constant([[[1, 0, 0, 0, 0, 0],
        [-1, -1, -1, 0, 0, -1]],
        [[0, 1, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0],
        [-1, 0, -1, 0, 0, 0]],
        [[0, 0, 0, 1, 0, 0],
        [0, 0, 0, -1, -1, 0]],
        [[0, 0, 0, 0, 1, 0],
        [-1, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 1],
        [0, 0, 0, -1, 0, 0]]])
        transition_kernels = K.permute_dimensions(transition_kernels, (1, 2, 0))
        transition_kernels = K.expand_dims(transition_kernels, -2)

        def custom_loss_dssp(y_true, y_pred):
            y_pred_one_hot = K.one_hot(K.argmax(y_pred), 6)
            def conv_loss(pred):
                return K.max(K.clip(K.conv2d(K.expand_dims(y_pred_one_hot, -1), transition_kernels), 0.0, 1.0), axis=-1)
            return (K.mean(losses.categorical_crossentropy(y_true, y_pred))
                    # inner max is over filters, which is important to only pick the most-activated filter at each site -
                    # this will be the filter that matches the identity of the DSSP code.
                    + 0.8 * K.mean(conv_loss(y_pred_one_hot))
                    + 0.4 * K.max(conv_loss(y_pred_one_hot))
                    + 0.4 * K.mean(conv_loss(y_pred))
                    + 0.2 * K.max(conv_loss(y_pred))
                   )

        def coeff_determination(y_true, y_pred):
            SS_res = K.sum(K.square(y_true - y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return 1 - SS_res/(SS_tot + K.epsilon())
    
        def custom_loss_stability(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) - 3.0 * coeff_determination(y_true, y_pred)
        
        loss = {
            "model_dssp": custom_loss_dssp,
            "model_stability": custom_loss_stability,
        }
        loss_weights = {"model_stability": 0.1, "model_dssp": 0.05}
        self.model.load_weights(checkpoint_filepath)
        self.model.compile(optimizer='adadelta', loss=loss, loss_weights=loss_weights)
        self.model.fit_generator(
                                 data_gen(self.batch_size),
                                 epochs=self.epochs,
                                 steps_per_epoch=1*X.shape[0]/self.batch_size,
                                 validation_data=(Xv, yv),
                                 callbacks=callbacks_list,
                                 verbose=self.verbose
                                )
        self.model.load_weights(checkpoint_filepath)

        os.remove(checkpoint_filepath)

    def _predict(self, X):
        return self.model.predict(np.expand_dims(np.stack([x for x in X.values]), 3))


def sequence_only_cnn_v2(max_residues, padding):
    amino_inputs = Input(shape=(23, max_residues + 2 + 2 * padding, 1))  # 20 amino acids plus null/beginning/end

    amino_model = Conv2D(400, (23, 5), kernel_regularizer=l2(.0), activation='relu')(amino_inputs)
    amino_model = Dropout(0.3)(amino_model)
    amino_model = Conv2D(200, (1, 9), kernel_regularizer=l2(.0), activation='relu')(amino_model)
    amino_model = Dropout(0.3)(amino_model)
    amino_model = Conv2D(100, (1, 17), kernel_regularizer=l2(.0), activation='relu')(amino_model)
    amino_model = Dropout(0.3)(amino_model)

    model = Flatten()(amino_model)

    model_dssp = Dense((max_residues + 2 + 2 * padding)*6)(model)
    model_dssp = Reshape(((max_residues + 2 + 2 * padding), 6))(model_dssp)
    model_dssp = Activation('softmax', name='model_dssp')(model_dssp) #softmax default axis is last axis
    model_dssp_flat = Flatten()(model_dssp)
    model = Concatenate()([model, model_dssp_flat])

    model = Dense(80, activation='elu', kernel_regularizer=l2(.0))(model)
    model = Dense(40, activation='elu', kernel_regularizer=l2(.0))(model)
    model = Dense(2, activation='linear', kernel_regularizer=l2(.0))(model)
    model_stability = Lambda(lambda x: K.concatenate([x, K.min(x, axis=1, keepdims=True)], axis=1), name='model_stability')(model)
    comp_model = Model(inputs=amino_inputs, outputs=[model_stability, model_dssp])

    # Based on permissible transitions between DSSP codes
    transition_kernels = K.constant([[[1, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, -1]],
    [[0, 1, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0]],
    [[0, 0, 1, 0, 0, 0],
    [-1, 0, -1, 0, 0, 0]],
    [[0, 0, 0, 1, 0, 0],
    [0, 0, 0, -1, -1, 0]],
    [[0, 0, 0, 0, 1, 0],
    [-1, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 1],
    [0, 0, 0, -1, 0, 0]]])
    transition_kernels = K.permute_dimensions(transition_kernels, (1, 2, 0))
    transition_kernels = K.expand_dims(transition_kernels, -2)

    def custom_loss_dssp(y_true, y_pred):
        y_pred_one_hot = K.one_hot(K.argmax(y_pred), 6)
        def conv_loss(pred):
            return K.max(K.clip(K.conv2d(K.expand_dims(y_pred_one_hot, -1), transition_kernels), 0.0, 1.0), axis=-1)
        return (K.mean(losses.categorical_crossentropy(y_true, y_pred))
                # inner max is over filters, which is important to only pick the most-activated filter at each site -
                # this will be the filter that matches the identity of the DSSP code.
                + 0.8 * K.mean(conv_loss(y_pred_one_hot))
                + 0.4 * K.max(conv_loss(y_pred_one_hot))
                + 0.4 * K.mean(conv_loss(y_pred))
                + 0.2 * K.max(conv_loss(y_pred))
               )

    def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res/(SS_tot + K.epsilon())
    
    def custom_loss_stability(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) - 3.0 * coeff_determination(y_true, y_pred)
        
    loss = {
        "model_dssp": custom_loss_dssp,
        "model_stability": custom_loss_stability,
    }
    loss_weights = {"model_stability": 0.2, "model_dssp": 1.2}
    comp_model.compile(optimizer='adadelta', loss=loss, loss_weights=loss_weights)

    mr = KerasRegressionTwoDimensional(model=model, model_description='Sequence CNN v2 regressor: 400x5->200x9->100x17->80->40->1',
                                       batch_size=128, epochs=50)
    return mr
