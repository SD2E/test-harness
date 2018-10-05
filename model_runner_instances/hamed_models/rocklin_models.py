import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from test_harness.model_runner_subclasses.mr_sklearn_classification import SklearnClassification
from test_harness.model_runner_subclasses.mr_sklearn_regression import SklearnRegression

rocklins_features = ['avg_all_frags', 'net_atr_net_sol_per_res', 'n_charged', 'buried_np_afilmvwy_per_res',
                     'avg_best_frag', 'fa_atr_per_res', 'exposed_polars', 'unsat_hbond',
                     'mismatch_probability', 'hbond_lr_bb', 'exposed_np_afilmvwy', 'fa_rep_per_res',
                     'degree', 'p_aa_pp', 'netcharge', 'worstfrag', 'frac_sheet', 'buried_np_per_res',
                     'abego_res_profile_penalty', 'hbond_sc', 'holes', 'cavity_volume', 'score_per_res',
                     'hydrophobicity', 'hbond_bb_sc', 'ss_sc', 'contig_not_hp_max', 'contact_all', 'omega',
                     'exposed_hydrophobics', 'contig_not_hp_avg']

rocklins_EHEE_features = rocklins_features + ['abd50_mean', 'abd50_min', 'dsc50_mean', 'dsc50_min',
                                              'ssc50_mean', 'ssc50_min']
rocklins_cols_to_use_per_topology_dict = {'HHH': rocklins_features, 'EHEE': rocklins_EHEE_features,
                                          'HEEH': rocklins_features, 'EEHEE': rocklins_features}


def rocklins_logistic_classifier_topology_general(train=None, test=None):
    rocklins_logistic_model = LogisticRegression(penalty='l1', C=0.1)

    mr = SklearnClassification(model=rocklins_logistic_model,
                               model_description="Rocklin Logistic: sklearn LogisticRegression with penalty='l1' and C=0.1",
                               feature_cols_to_use=rocklins_features, topology_specific_or_general='general',
                               training_data=train, testing_data=test, data_set_description='15k',
                               train_test_split_description='12k-3k',
                               predict_untested=False
                               )

    return mr


def rocklins_logistic_classifier_topology_specific():
    rocklins_logistic_model = LogisticRegression(penalty='l1', C=0.1)

    mr = SklearnClassification(model=rocklins_logistic_model,
                               model_description="Rocklin Logistic: sklearn LogisticRegression with penalty='l1' and C=0.1",
                               feature_cols_to_use=rocklins_cols_to_use_per_topology_dict,
                               topology_specific_or_general='specific')

    return mr


def rocklins_linear_regression_topology_general(train, test):
    rocklins_linear_model = LinearRegression()

    mr = SklearnRegression(model=rocklins_linear_model,
                           model_description='Rocklin LinReg: Default sklearn linear regression',
                           feature_cols_to_use=rocklins_features, topology_specific_or_general='general',
                           training_data=train, testing_data=test, data_set_description='15k',
                           train_test_split_description='12k-3k',
                           predict_untested=False
                           )

    return mr


def rocklins_linear_regression_topology_specific():
    rocklins_linear_model = LinearRegression()

    mr = SklearnRegression(model=rocklins_linear_model,
                           model_description='Rocklin LinReg: Default sklearn linear regression',
                           feature_cols_to_use=rocklins_cols_to_use_per_topology_dict,
                           topology_specific_or_general='specific')

    return mr


def rocklins_gradboost_regression_topology_general():
    rocklins_gradboost_model = GradientBoostingRegressor(n_estimators=250, max_depth=5, min_samples_split=5,
                                                         learning_rate=0.01, loss='ls')
    mr = SklearnRegression(model=rocklins_gradboost_model,
                           model_description="""Rocklin GradBoost: sklearn GradientBoostingRegressor with n_estimators=250, max_depth=5, min_samples_split=5, learning_rate=0.01, and loss='ls'""",
                           feature_cols_to_use=rocklins_features, topology_specific_or_general='general')

    return mr


def rocklins_gradboost_regression_topology_specific():
    rocklins_gradboost_model = GradientBoostingRegressor(n_estimators=250, max_depth=5, min_samples_split=5,
                                                         learning_rate=0.01, loss='ls')
    mr = SklearnRegression(model=rocklins_gradboost_model,
                           model_description="""Rocklin GradBoost: sklearn GradientBoostingRegressor with n_estimators=250, max_depth=5, min_samples_split=5, learning_rate=0.01, and loss='ls'""",
                           feature_cols_to_use=rocklins_cols_to_use_per_topology_dict,
                           topology_specific_or_general='specific')

    return mr


def linear_regression_topology_general_all_features(training_data, testing_data, col_to_predict, data_set_description,
                                                    train_test_split_description):
    rocklins_linear_model = LinearRegression()
    mr = SklearnRegression(model=rocklins_linear_model,
                           model_description='Baseline Linear Regression',
                           topology_specific_or_general='general',
                           training_data=training_data.copy(), testing_data=testing_data.copy(),
                           data_set_description=data_set_description,
                           train_test_split_description=train_test_split_description,
                           predict_untested=False, col_to_predict=col_to_predict,
                           )
    return mr


def logistic_classifier_topology_general_all_features(training_data, testing_data, col_to_predict, data_set_description,
                                                      train_test_split_description):
    rocklins_logistic_model = LogisticRegression(penalty='l1', C=0.1)

    mr = SklearnClassification(model=rocklins_logistic_model,
                               model_description="Rocklin Logistic: sklearn LogisticRegression with penalty='l1' and C=0.1",
                               col_to_predict=col_to_predict, topology_specific_or_general='general',
                               predict_untested=False,
                               training_data=training_data.copy(), testing_data=testing_data.copy(),
                               train_test_split_description=train_test_split_description,
                               data_set_description=data_set_description)

    return mr


def linear_regression_topology_general_all_features_diff_train_test():
    default_data_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        'model_runner_data/default_model_runner_data/')
    print(default_data_folder_path)
    train_path = os.path.join(default_data_folder_path, 'consistent_normalized_training_data_v1.csv')
    test_path = os.path.join(default_data_folder_path, 'consistent_normalized_testing_data_v1.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    combined = pd.concat([train, test])
    train = combined.loc[combined['library'] == 'Rocklin']
    test = combined.loc[combined['library'] != 'Rocklin']

    rocklins_linear_model = LinearRegression()

    mr = SklearnRegression(model=rocklins_linear_model,
                           model_description='Rocklin LinReg: Default sklearn linear regression',
                           topology_specific_or_general='general', training_data=train,
                           testing_data=test, data_set_description='85k data',
                           train_test_split_description="train = Rocklin 15k, test = 70k")

    return mr


def logistic_classifier_topology_general_all_features_diff_train_test():
    default_data_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        'model_runner_data/default_model_runner_data/')
    print(default_data_folder_path)
    train_path = os.path.join(default_data_folder_path, 'consistent_normalized_training_data_v1.csv')
    test_path = os.path.join(default_data_folder_path, 'consistent_normalized_testing_data_v1.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    combined = pd.concat([train, test])
    train = combined.loc[combined['library'] == 'Rocklin']
    test = combined.loc[combined['library'] != 'Rocklin']

    rocklins_logistic_model = LogisticRegression(penalty='l1', C=0.1)

    mr = SklearnClassification(model=rocklins_logistic_model,
                               model_description="Rocklin Logistic: sklearn LogisticRegression with penalty='l1' and C=0.1",
                               topology_specific_or_general='general', training_data=train,
                               testing_data=test, data_set_description='85k data',
                               train_test_split_description="train = Rocklin 15k, test = 70k")

    return mr
