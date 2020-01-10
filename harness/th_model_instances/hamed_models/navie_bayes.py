from sklearn.naive_bayes import GaussianNB
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def gaussan_naive_bayes_classification(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight="balanced"):
    # Creating an sklearn random forest classification model:
    gnb = GaussianNB()

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=gnb, model_author='Mohammed',
                                     model_description="Gaussan Naive Bayes")
    return th_model