from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.weighted_logistic import weighted_logistic_classifier
from harness.th_model_instances.perovskite_models.zhi_svm import support_vector_radial_basis_classifier
from harness.th_model_instances.baseline_models.classification_baselines import baseline_uniform
from harness.th_model_instances.perovskite_models.xgboost import gradient_boosted_tree
from harness.th_model_instances.perovskite_models.rxn_only import rxn_only_svm
from harness.th_model_instances.perovskite_models.rxn_ratio_only import rxn_intuition_svm

MODELS_TO_RUN = \
    [
        baseline_uniform,
        weighted_logistic_classifier,
        rxn_only_svm,
        rxn_intuition_svm,
        random_forest_classification,
        gradient_boosted_tree,        
        support_vector_radial_basis_classifier,
    ]
