import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from test_harness.model_runner_subclasses.mr_sklearn_classification import SklearnClassification


def random_forest_classification():
    # Creating an sklearn random forest classification model:
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1)
    # Creating an instance of the SklearnClassification Model Runner subclass
    mr_rfc = SklearnClassification(model=rfc,
                                   model_description="Random Forest: n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13, n_jobs=-1",
                                   col_to_predict='stable?', topology_specific_or_general='general')
    return mr_rfc


def random_forest_classification_diff_train_test():
    default_data_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        'model_runner_data/default_model_runner_data/')
    train_path = os.path.join(default_data_folder_path, 'consistent_normalized_training_data_v1.csv')
    test_path = os.path.join(default_data_folder_path, 'consistent_normalized_testing_data_v1.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    combined = pd.concat([train, test])
    train = combined.loc[combined['library'] == 'Rocklin']
    test = combined.loc[combined['library'] != 'Rocklin']

    # Creating an sklearn random forest classification model:
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1)
    # Creating an instance of the SklearnClassification Model Runner subclass
    mr_rfc = SklearnClassification(model=rfc,
                                   model_description="Random Forest: n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13, n_jobs=-1",
                                   col_to_predict='stable?', topology_specific_or_general='general',
                                   training_data=train, testing_data=test, data_set_description='85k data',
                                   train_test_split_description="train = Rocklin 15k, test = 70k")
    return mr_rfc
