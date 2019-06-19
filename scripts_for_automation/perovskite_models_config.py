import numpy as np

from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.weighted_logistic import weighted_logistic_classifier
from harness.th_model_instances.perovskite_models.zhi_svm import support_vector_radial_basis_classifier
from harness.th_model_instances.baseline_models.classification_baselines import baseline_uniform
from harness.th_model_instances.perovskite_models.xgboost import gradient_boosted_tree
from harness.th_model_instances.perovskite_models.rxn_only import rxn_only_svm, rxn_intuition_svm


def add_ratio_features(df):
    df['_feat_org_inorg_ratio'] = (np.arctan2(df._rxn_M_organic, df._rxn_M_inorganic) - np.pi / 4) ** 2
    df['_feat_org_inorg_2ratio'] = (np.arctan2(df._rxn_M_organic, df._rxn_M_inorganic) - 3 * np.pi / 8) ** 2
    return df


rxn_only_features = ['_rxn_M_acid', '_rxn_M_inorganic', '_rxn_M_organic', '_rxn_mixingtime1S',
                     '_rxn_mixingtime2S', '_rxn_reactiontimeS', '_rxn_stirrateRPM']

rxn_ratio_features = ['_rxn_M_acid', '_rxn_M_inorganic', '_rxn_M_organic',
                      # '_feat_org_inorg_2ratio',
                      '_feat_org_inorg_ratio',
                      ]

MODELS_TO_RUN = \
    {
        baseline_uniform: {"features_to_use": None},
        weighted_logistic_classifier: {"features_to_use": None},
        rxn_only_svm: {"features_to_use": rxn_only_features},
        rxn_intuition_svm: {"features_to_use": rxn_ratio_features},
        random_forest_classification: {"features_to_use": None},
        gradient_boosted_tree: {"features_to_use": None},
        support_vector_radial_basis_classifier: {"features_to_use": None},
    }
