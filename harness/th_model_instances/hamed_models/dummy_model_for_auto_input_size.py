import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from harness.th_model_classes.class_sklearn_regression import SklearnRegression
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def dummy_regressor():
    rfr = RandomForestRegressor()
    return SklearnRegression(model=rfr, model_author="", model_description="dummy_regressor")


def dummy_classifier():
    rfc = RandomForestClassifier()
    return SklearnClassification(model=rfc, model_author="", model_description="dummy_classifier")
