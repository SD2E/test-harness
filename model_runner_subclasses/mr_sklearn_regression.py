from test_harness.model_runner_base_classes import RegressionModelRunner


class SklearnRegression(RegressionModelRunner):
    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)
