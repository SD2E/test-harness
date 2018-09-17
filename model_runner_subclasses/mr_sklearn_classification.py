from test_harness.model_runner_base_classes import ClassificationModelRunner


class SklearnClassification(ClassificationModelRunner):
    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
