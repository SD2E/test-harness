import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from test_harness.model_runner_subclasses.mr_sklearn_classification import SklearnClassification


def random_forest_classification(training_data, testing_data, col_to_predict, data_set_description,
                                 train_test_split_description):
    # Creating an sklearn random forest classification model:
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight="balanced")
    # Creating an instance of the SklearnClassification Model Runner subclass
    mr_rfc = SklearnClassification(model=rfc,
                                   model_description="Random Forest: n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13, n_jobs=-1",
                                   col_to_predict=col_to_predict,
                                   predict_untested=False,
                                   training_data=training_data.copy(), testing_data=testing_data.copy(),
                                   train_test_split_description=train_test_split_description,
                                   data_set_description=data_set_description)
    return mr_rfc