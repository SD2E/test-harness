from harness.test_harness_models_abstract_classes import ClassificationModel


class SklearnClassification(ClassificationModel):
    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _predict_proba(self, X):
        # this should return the probability of being in class 1 (positive class)
        return self.model.predict_proba(X)[:, 1]
