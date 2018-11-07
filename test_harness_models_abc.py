import inspect
from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression


class TestHarnessModel(metaclass=ABCMeta):
    def __init__(self, model, model_description):
        self.model = model
        self.model_description = model_description
        self.stack_trace = inspect.stack()

    @abstractmethod
    def _fit(self, X_train, y_train):
        pass

    @abstractmethod
    def _predict(self, X_test):
        pass


class ClassificationModel(TestHarnessModel, metaclass=ABCMeta):
    def __init__(self, model=LogisticRegression(), model_description='Default Sklearn Logistic Classifier'):
        super().__init__(model, model_description)

    # in subclasses, this method should return probability values for being in the positive (1) class
    @abstractmethod
    def _predict_proba(self, X_test):
        pass


class RegressionModel(TestHarnessModel, metaclass=ABCMeta):
    def __init__(self, model=LinearRegression(), model_description='Default Sklearn Linear Regression'):
        super().__init__(model, model_description)
