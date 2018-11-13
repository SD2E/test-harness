from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from test_harness.th_model_classes.class_keras_regression import KerasRegression


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def keras_regression_1():
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.0, input_shape=(110,)))
    kr.add(Dense(units=9, activation="relu"))
    kr.add(Dense(units=61, activation="relu"))
    kr.add(Dense(units=1))
    kr.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression Model Runner subclass
    mr_kr = KerasRegression(model=kr,
                            model_description='Keras: 2 hidden layers (9 and 61 nodes)', col_to_predict='stabilityscore'
                            )

    return mr_kr


def keras_regression_2():
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.0, input_shape=(110,)))
    kr.add(Dense(units=9, activation="relu"))
    kr.add(Dense(units=87, activation="relu"))
    kr.add(Dense(units=1))
    kr.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression Model Runner subclass
    mr_kr = KerasRegression(model=kr,
                            model_description='Keras: 2 hidden layers (9 and 87 nodes)',
                            col_to_predict='stabilityscore')

    return mr_kr


def keras_regression_3():
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.0, input_shape=(110,)))
    kr.add(Dense(units=89, activation="relu"))
    kr.add(Dense(units=93, activation="relu"))
    kr.add(Dense(units=1))
    kr.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression Model Runner subclass
    mr_kr = KerasRegression(model=kr,
                            model_description='Keras: 2 hidden layers (89 and 93 nodes)',
                            col_to_predict='stabilityscore',
                            )

    return mr_kr


def keras_regression_4():
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.20641837312160777, input_shape=(110,)))
    kr.add(Dense(units=100, activation="relu"))
    kr.add(Dropout(0.7))
    kr.add(Dense(units=20, activation="relu"))
    kr.add(Dropout(0.7))
    kr.add(Dense(units=1))
    kr.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression Model Runner subclass
    mr_kr = KerasRegression(model=kr,
                            model_description='Keras: 2 hidden layers (100 and 20 nodes)',
                            col_to_predict='stabilityscore')

    return mr_kr


def keras_regression_5a():
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.0, input_shape=(110,)))
    kr.add(Dense(units=80, activation="relu"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=73, activation="relu"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=1))
    kr.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression Model Runner subclass
    mr_kr = KerasRegression(model=kr,
                            model_description='Keras: 2 hidden layers (80 and 73 nodes) - Relu',
                            col_to_predict='stabilityscore')

    return mr_kr


def keras_regression_5b():
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.0, input_shape=(110,)))
    kr.add(Dense(units=80, activation="softplus"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=73, activation="softplus"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=1))
    kr.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression Model Runner subclass
    mr_kr = KerasRegression(model=kr,
                            model_description='Keras: 2 hidden layers (80 and 73 nodes) - Softplus',
                            col_to_predict='stabilityscore')

    return mr_kr


def keras_regression_best(train=None, test=None):
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.0, input_shape=(110,)))
    kr.add(Dense(units=80, activation="relu"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=73, activation="relu"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=1))
    kr.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression Model Runner subclass
    mr_kr = KerasRegression(model=kr,
                            model_description='Keras: 2 hidden layers (80 and 73 nodes)',
                            col_to_predict='stabilityscore',
                            training_data=train, testing_data=test, data_set_description='15k',
                            train_test_split_description='12k-3k',
                            predict_untested=False
                            )

    return mr_kr


def jeds_keras_regression():
    kr = Sequential()
    kr.add(Dense(80, activation='softplus', kernel_regularizer=l2(.01), input_shape=(110,)))
    kr.add(Dense(40, activation='softplus', kernel_regularizer=l2(.01)))
    kr.add(Dense(1, activation='linear', kernel_regularizer=l2(0.01)))
    kr.compile(optimizer='adam', loss=root_mean_squared_error)

    mr_kr = KerasRegression(model=kr,
                            model_description="Jed's Keras Model: 2 hidden layers (80 and 40 nodes)', "
                                              "col_to_predict='stabilityscore")

    return mr_kr
