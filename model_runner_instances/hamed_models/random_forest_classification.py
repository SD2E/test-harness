from sklearn.ensemble import RandomForestClassifier
from test_harness.model_runner_subclasses.mr_sklearn_classification import SklearnClassification


def random_forest_classification():
    # Creating an sklearn random forest classification model:
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight="balanced")

    # Creating an instance of the SklearnClassification Model Runner subclass
    mr_rfc = SklearnClassification(model=rfc,
                                   model_description="Random Forest: n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13, n_jobs=-1")
    return mr_rfc