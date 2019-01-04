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
    mr_kr = KerasRegression(model=kr, model_description='Keras: 2 hidden layers (9 and 61 nodes)')

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
    mr_kr = KerasRegression(model=kr, model_description='Keras: 2 hidden layers (9 and 87 nodes)')

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
    mr_kr = KerasRegression(model=kr, model_description='Keras: 2 hidden layers (89 and 93 nodes)')

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
    mr_kr = KerasRegression(model=kr, model_description='Keras: 2 hidden layers (100 and 20 nodes)')

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
    mr_kr = KerasRegression(model=kr, model_description='Keras: 2 hidden layers (80 and 73 nodes) - Relu')

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
    mr_kr = KerasRegression(model=kr, model_description='Keras: 2 hidden layers (80 and 73 nodes) - Softplus')

    return mr_kr


def keras_regression_best():
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
    mr_kr = KerasRegression(model=kr, model_description='Keras: 2 hidden layers (80 and 73 nodes)')

    return mr_kr
