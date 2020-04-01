from sklearn.naive_bayes import GaussianNB
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def gaussian_naive_bayes_classification():
    '''

    :param n_estimators:
    :param max_features:
    :param criterion:
    :param min_samples_leaf:
    :param n_jobs:
    :param class_weight:
    :return:
    '''
    # Creating an sklearn random forest classification model:
    gnb = GaussianNB()

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=gnb, model_author='Mohammed',
                                     model_description="Gaussan Naive Bayes")
    return th_model