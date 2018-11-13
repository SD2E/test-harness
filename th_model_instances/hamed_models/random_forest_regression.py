import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from test_harness.th_model_classes.class_sklearn_regression import SklearnRegression

from test_harness.th_model_classes.class_rf_features import RFRegression


def rfr_features(training_data, testing_data, col_to_predict, data_set_description, train_test_split_description):
    # Creating an sklearn random forest regression model:
    rfr = RandomForestRegressor(bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2,
                                max_features=0.2, max_depth=86, n_jobs=-1)
    # Creating an instance of the SklearnRegression Model Runner subclass
    mr_rfr = RFRegression(model=rfr,
                          model_description="Random Forest Regressor: bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2, max_features=0.2, max_depth=86, n_jobs=-1",
                          col_to_predict=col_to_predict,
                          predict_untested=False,
                          training_data=training_data.copy(), testing_data=testing_data.copy(),
                          train_test_split_description=train_test_split_description,
                          data_set_description=data_set_description)
    return mr_rfr


def random_forest_regression(train, test):
    # Creating an sklearn random forest regression model:
    rfr = RandomForestRegressor(bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2,
                                max_features=0.2, max_depth=86, n_jobs=-1)
    # Creating an instance of the SklearnRegression Model Runner subclass
    mr_rfr = SklearnRegression(model=rfr,
                               model_description="Random Forest Regressor: bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2, max_features=0.2, max_depth=86, n_jobs=-1",
                               col_to_predict='stabilityscore',
                               predict_untested=False,
                               training_data=train, testing_data=test, train_test_split_description='',
                               data_set_description='')
    return mr_rfr


def random_forest_regression_less_features():
    features = ['avg_all_frags', 'avg_best_frag', 'sum_best_frags', 'buried_np_afilmvwy_per_res', 'worst6frags',
                'abego_res_profile_penalty', 'score_per_res', 'hydrophobicity', 'fa_atr_per_res', 'worstfrag',
                'contig_not_hp_avg', 'abego_res_profile', 'net_atr_per_res', 'buried_np_afilmvwy', 'total_score',
                'exposed_np_afilmvwy', 'hxl_tors', 'hphob_sc_contacts', 'largest_hphob_cluster', 'fa_intra_atr_xover4',
                'buried_np_per_res', 'netcharge', 'net_atr_net_sol_per_res', 'ref', 'n_hydrophobic_noa',
                'net_sol_per_res', 'hbond_sr_bb', 'fa_intra_rep_xover4', 'fa_dun_semi', 'fa_intra_elec',
                'hbond_sr_bb_per_helix', 'p_aa_pp', 'helix_sc', 'hphob_sc_degree', 'fa_dun_rot', 'buried_np',
                'hbond_sc', 'omega', 'rama_prepro', 'lk_ball', 'exposed_total', 'fa_sol', 'contig_not_hp_avg_norm',
                'exposed_hydrophobics', 'fa_elec', 'chymo_with_lm_cut_sites', 'hbond_bb_sc', 'degree', 'holes',
                'fa_atr', 'lk_ball_iso', 'fa_rep_per_res', 'fa_dun_dev', 'fa_rep', 'fa_intra_sol_xover4']

    # Creating an sklearn random forest regression model:
    rfr = RandomForestRegressor(bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2,
                                max_features=0.2, max_depth=86, n_jobs=-1)
    # Creating an instance of the SklearnRegression Model Runner subclass
    mr_rfr = SklearnRegression(model=rfr,
                               model_description="Random Forest Regressor: bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2, max_features=0.2, max_depth=86, n_jobs=-1",
                               col_to_predict='stabilityscore',
                               feature_cols_to_use=features)
    return mr_rfr


def random_forest_regression_diff_train_test():
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

    # Creating an sklearn random forest regression model:
    rfr = RandomForestRegressor(bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2,
                                max_features=0.2, max_depth=86, n_jobs=-1)
    # Creating an instance of the SklearnRegression Model Runner subclass
    mr_rfr = SklearnRegression(model=rfr,
                               model_description="Random Forest Regressor: bootstrap=False, min_samples_leaf=1, n_estimators=689, min_samples_split=2, max_features=0.2, max_depth=86, n_jobs=-1",
                               col_to_predict='stabilityscore',
                               training_data=train,
                               testing_data=test, data_set_description='85k data',
                               train_test_split_description="train = Rocklin 15k, test = 70k")
    return mr_rfr
