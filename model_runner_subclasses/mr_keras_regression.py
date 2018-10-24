from test_harness.test_harness_models_abc import RegressionModel


class KerasRegression(RegressionModel):
    def __init__(self, model, model_description, training_data=None, testing_data=None,
                 data_set_description=None, train_test_split_description=None, col_to_predict='stabilityscore',
                 feature_cols_to_use=None, id_col='name', topology_col='topology', predict_untested=None, epochs=25, batch_size=1000,
                 verbose=0):
        super(KerasRegression, self).__init__(model, model_description, training_data, testing_data,
                                              data_set_description, train_test_split_description, col_to_predict,
                                              feature_cols_to_use, id_col, topology_col,
                                              predict_untested)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def _predict(self, X):
        return self.model.predict(X)
