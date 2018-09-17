from test_harness.model_runner_base_classes import RegressionModelRunner


class KerasRegression(RegressionModelRunner):
    def __init__(self, model, model_description, training_data=None, testing_data=None,
                 data_set_description=None, train_test_split_description=None, col_to_predict='stabilityscore',
                 feature_cols_to_use=None, id_col='name', topology_col='topology',
                 topology_specific_or_general='general', predict_untested=None, cv=False, epochs=25, batch_size=1000,
                 verbose=0):
        super(KerasRegression, self).__init__(model, model_description, training_data, testing_data,
                                              data_set_description, train_test_split_description, col_to_predict,
                                              feature_cols_to_use, id_col, topology_col,
                                              topology_specific_or_general, predict_untested)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def _predict(self, X):
        return self.model.predict(X)
