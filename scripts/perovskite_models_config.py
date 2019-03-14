from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.weighted_logistic import weighted_logistic_classifier
from harness.th_model_instances.perovskite_models.zhi_svm import support_vector_radial_basis_classifier

MODELS_TO_RUN = \
    [
        support_vector_radial_basis_classifier,
        random_forest_classification,
        weighted_logistic_classifier
    ]
