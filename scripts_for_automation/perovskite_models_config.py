from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.weighted_logistic import weighted_logistic_classifier


def models_to_run():
    model_dict = [random_forest_classification, weighted_logistic_classifier]
    return model_dict
