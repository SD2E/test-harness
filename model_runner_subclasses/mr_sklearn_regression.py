from test_harness.test_harness_models_abstract_classes import RegressionModel


class SklearnRegression(RegressionModel):
    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)
