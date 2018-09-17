import os
import pandas as pd
from keras.optimizers import SGD
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import Dense, Dropout
from test_harness.model_runner_subclasses.mr_keras_classification import KerasClassification


def keras_classification_1(train=None, test=None):
    # Creating a Keras deep learning classification model:
    model = Sequential()
    model.add(Dropout(0.0, input_shape=(110,)))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(units=85, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.3081636734334996))
    model.add(Dense(units=49, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.3081636734334996))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.065883686787412021, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 4.914205344585091
    class_weights = {0: 1, 1: imbalance_ratio}

    # Creating an instance of the KerasClassification Model Runner subclass
    mr_kc = KerasClassification(model=model,
                                model_description='Keras: 2 hidden layers (85 and 49 nodes), weighted, dropout=0.308',
                                col_to_predict='stable?', feature_cols_to_use=None,
                                topology_specific_or_general='general', class_weight=class_weights,
                                training_data=train, testing_data=test, data_set_description='15k',
                                train_test_split_description='12k-3k',
                                predict_untested=False
                                )
    return mr_kc


def keras_classification_1_diff_train_test():
    # Creating a Keras deep learning classification model:
    model = Sequential()
    model.add(Dropout(0.0, input_shape=(110,)))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(units=85, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.3081636734334996))
    model.add(Dense(units=49, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.3081636734334996))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.065883686787412021, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 4.914205344585091
    class_weights = {0: 1, 1: imbalance_ratio}
    default_data_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        'model_runner_data/default_model_runner_data/')
    train_path = os.path.join(default_data_folder_path, 'consistent_normalized_training_data_v1.csv')
    test_path = os.path.join(default_data_folder_path, 'consistent_normalized_testing_data_v1.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    combined = pd.concat([train, test])
    train = combined.loc[combined['library'] == 'Rocklin']
    test = combined.loc[combined['library'] != 'Rocklin']

    # Creating an instance of the KerasClassification Model Runner subclass
    mr_kc = KerasClassification(model=model,
                                model_description='Keras: 2 hidden layers (85 and 49 nodes), weighted, dropout=0.308',
                                col_to_predict='stable?', feature_cols_to_use=None,
                                topology_specific_or_general='general', class_weight=class_weights, training_data=train,
                                testing_data=test, data_set_description='85k data',
                                train_test_split_description="train = Rocklin 15k, test = 70k")
    return mr_kc


def keras_classification_2():
    model = Sequential()
    model.add(Dropout(0.0, input_shape=(110,)))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(units=7, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.69999999999999996))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.20000000000000001, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 5.983771569433032
    class_weights = {0: 1, 1: imbalance_ratio}

    mr_kc = KerasClassification(model=model,
                                model_description='Keras: 1 hidden layer (7 nodes), weighted 5.984, dropout=0.699, lr=0.2, l2=0',
                                col_to_predict='stable?', feature_cols_to_use=None,
                                topology_specific_or_general='general', class_weight=class_weights)
    return mr_kc


def keras_classification_3():
    model = Sequential()
    model.add(Dropout(0.45332796312869916, input_shape=(110,)))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(units=23, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.0))
    wr = l1_l2(l2=0, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.20000000000000001, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 5.983771569433032
    class_weights = {0: 1, 1: imbalance_ratio}

    mr_kc = KerasClassification(model=model,
                                model_description='Keras: 1 hidden layer (23 nodes), weighted 5.984, dropout_in=0.453, dropout=0, lr=0.2, l2=0',
                                col_to_predict='stable?', feature_cols_to_use=None,
                                topology_specific_or_general='general', class_weight=class_weights)
    return mr_kc


def keras_classification_4():
    # Creating a Keras deep learning classification model:
    model = Sequential()
    model.add(Dropout(0.12798022511149154, input_shape=(110,)))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(units=64, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    model.add(Dense(units=55, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.18012050376588148, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 5.983771569433032
    class_weights = {0: 1, 1: imbalance_ratio}

    # Creating an instance of the KerasClassification Model Runner subclass
    mr_kc = KerasClassification(model=model,
                                model_description='Keras: 2 hidden layers (64 and 55 nodes), weighted 5.984, dropout_in=0.128, dropout=0.35, lr=0.18, l2=0.0018',
                                col_to_predict='stable?', feature_cols_to_use=None,
                                topology_specific_or_general='general', class_weight=class_weights)
    return mr_kc
