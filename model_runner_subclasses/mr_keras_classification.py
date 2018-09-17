from test_harness.model_runner_base_classes import ClassificationModelRunner


class KerasClassification(ClassificationModelRunner):
    def __init__(self, model, model_description, training_data=None, testing_data=None,
                 data_set_description=None, train_test_split_description=None, col_to_predict='stable?',
                 feature_cols_to_use=None, id_col='name', topology_col='topology',
                 topology_specific_or_general='general', predict_untested=None, cv=False, epochs=25, batch_size=1000,
                 class_weight=None, verbose=0):
        super(KerasClassification, self).__init__(model, model_description, training_data, testing_data,
                                                  data_set_description, train_test_split_description, col_to_predict,
                                                  feature_cols_to_use, id_col, topology_col,
                                                  topology_specific_or_general, predict_untested, cv)
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.verbose = verbose

    def _fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, class_weight=self.class_weight,
                       verbose=self.verbose)

    def _predict(self, X):
        return self.model.predict_classes(X)

    def _predict_proba(self, X):
        probs = [x[0] for x in self.model.predict(X)]
        return probs
