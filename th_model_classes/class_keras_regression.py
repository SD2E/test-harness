from test_harness.test_harness_models_abstract_classes import RegressionModel


class KerasRegression(RegressionModel):
    def __init__(self, model, model_description, epochs=25, batch_size=1000, verbose=0):
        super(KerasRegression, self).__init__(model, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def _predict(self, X):
        return self.model.predict(X)
