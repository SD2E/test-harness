from sklearn.dummy import DummyClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


# think about: should I make the baseline models a single function with strategy as an argument?

def baseline_prior():
    strategy = 'prior'
    baseline_model = DummyClassifier(strategy=strategy, random_state=5)

    th_model = SklearnClassification(model=baseline_model, model_author="Hamed",
                                     model_description="Baseline DummyClassifier with strategy='{}'".format(strategy))
    return th_model


def baseline_stratified():
    strategy = 'stratified'
    baseline_model = DummyClassifier(strategy=strategy, random_state=5)

    th_model = SklearnClassification(model=baseline_model, model_author="Hamed",
                                     model_description="Baseline DummyClassifier with strategy='{}'".format(strategy))
    return th_model


def baseline_uniform():
    strategy = 'uniform'
    baseline_model = DummyClassifier(strategy=strategy, random_state=5)

    th_model = SklearnClassification(model=baseline_model, model_author="Hamed",
                                     model_description="Baseline DummyClassifier with strategy='{}'".format(strategy))
    return th_model
