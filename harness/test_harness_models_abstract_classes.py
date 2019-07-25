import inspect
from abc import ABCMeta, abstractmethod


class TestHarnessModel(metaclass=ABCMeta):
    def __init__(self, model, model_author, model_description):
        self.model = model
        self.model_author = model_author
        self.model_description = model_description
        # this will get the name of the function that called the TestHarnessModel object.
        # E.g. see harness/th_model_instances/hamed_models/random_forest_regression.py
        self.model_name = inspect.stack()[1].function
        self.stack_trace = inspect.stack()

    @abstractmethod
    def _fit(self, X_train, y_train):
        pass

    @abstractmethod
    def _predict(self, X_test):
        pass


class ClassificationModel(TestHarnessModel, metaclass=ABCMeta):
    # in subclasses, this method should return probability values for being in the positive (1) class
    @abstractmethod
    def _predict_proba(self, X_test):
        pass


class RegressionModel(TestHarnessModel, metaclass=ABCMeta):
    pass
    # # the init is only here so the subclass isn't empty...
    # def __init__(self, model, model_author, model_description):
    #     super().__init__(model, model_author, model_description)
