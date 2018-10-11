from sklearn.linear_model import LogisticRegression
from test_harness.model_runner_subclasses.mr_sklearn_classification import SklearnClassification


def weighted_logistic_classifier(training_data, testing_data, col_to_predict, data_set_description,
                                 train_test_split_description):
    rocklins_logistic_model_with_weighted_classes_added = LogisticRegression(penalty='l1', C=0.1,
                                                                             class_weight="balanced")

    mr = SklearnClassification(model=rocklins_logistic_model_with_weighted_classes_added,
                               model_description="Rocklin Logistic: sklearn LogisticRegression with penalty='l1' and C=0.1",
                               col_to_predict=col_to_predict,
                               predict_untested=False,
                               training_data=training_data.copy(), testing_data=testing_data.copy(),
                               train_test_split_description=train_test_split_description,
                               data_set_description=data_set_description)

    return mr
