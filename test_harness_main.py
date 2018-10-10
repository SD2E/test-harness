import argparse
import datetime as dt
import os
import importlib
import types
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from sklearn.model_selection import train_test_split
from test_harness_class import TestHarness

from model_runner_instances.hamed_models.random_forest_regression import rfr_features
from model_runner_instances.hamed_models.random_forest_classification import random_forest_classification
from model_runner_instances.jed_models.sequence_cnn import sequence_only_cnn
from model_runner_instances.hamed_models.rocklin_models import linear_regression_topology_general_all_features as linreg
from model_runner_instances.hamed_models.rocklin_models import logistic_classifier_topology_general_all_features

# SET PATH TO DATA FOLDER IN LOCALLY CLONED `versioned-datasets` REPO HERE:
# Note that if you clone the `versioned-datasets` repo at the same level as where you cloned the `protein-design` repo,
# then you can use VERSIONED_DATASETS = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
print()

PWD = os.getcwd()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)

print("PWD:", PWD)
print("HERE:", HERE)
print("PARENT:", PARENT)
print()

parser = argparse.ArgumentParser()
# Default behavior is to write out relative
# to test_harness. Passing output will cause
# writes to occur to a path relative to the current working directory
parser.add_argument('--output', required=False,
                    help='Output directory')


def model_runner_by_name(model_runner_path,
                         module_base_path='model_runner_instances'):
    """
    Instantiate an instance of model_runner by path

    Returns: ModelRunner
    Raises: Exception
    """
    try:
        path_parts = model_runner_path.split('.')
        func_name = path_parts[-1]
        func_module_parent_path = module_base_path + '.' + '.'.join(path_parts[:-1])
        func_module = importlib.import_module(func_module_parent_path)
        named_meth = getattr(func_module, func_name)
        if callable(named_meth) and isinstance(named_meth, types.FunctionType):
            model_runner_instance = named_meth()
            return model_runner_instance
    # TODO: More granular Exception handling
    except Exception:
        raise


def main(args):
    model_list = []

    if 'output' in args and args.output is not None:
        output_dir = os.path.join(PWD, args.output)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception:
                raise
    else:
        output_dir = PARENT

    th = TestHarness(output_path=output_dir)

    combined_data = pd.read_csv(os.path.join(VERSIONED_DATA,
                                             'protein-design/aggregated_data/all_libs_cleaned.v1.aggregated_data.csv'),
                                comment='#', low_memory=False)

    combined_data['dataset_original'] = combined_data['dataset']
    combined_data['dataset'] = combined_data['dataset'].replace({"topology_mining_and_Longxing_chip_1": "t_l_untested",
                                                                 "topology_mining_and_Longxing_chip_2": "t_l_untested",
                                                                 "topology_mining_and_Longxing_chip_3": "t_l_untested"})

    # Changing the order of columns in combined_data
    col_order = list(combined_data.columns.values)
    col_order.insert(2, col_order.pop(col_order.index('dataset_original')))
    combined_data = combined_data[col_order]
    combined_data['stabilityscore_2classes'] = combined_data['stabilityscore'] > 1
    combined_data['stabilityscore_calibrated_2classes'] = combined_data['stabilityscore_calibrated'] > 1
    combined_data['stabilityscore_cnn_2classes'] = combined_data['stabilityscore_cnn'] > 1
    combined_data['stabilityscore_cnn_calibrated_2classes'] = combined_data['stabilityscore_cnn_calibrated'] > 1


    # do this later (need AUC multi-class alternative):
    # combined_data['3_stability_bins'] = pd.cut(combined_data['stabilityscore'], bins=[-100, 0, 1, 100],
    #                                            labels=["trash", "unstable", "stable"])
    # print(combined_data[['stabilityscore', '3_stability_bins']])

    data_RD_16k = combined_data.loc[combined_data['dataset_original'] == 'Rocklin'].copy()
    data_RD_BL_81k = combined_data.loc[
        combined_data['dataset_original'].isin(['Rocklin', 'Eva1', 'Eva2', 'Inna', 'Longxing'])].copy()
    data_RD_BL_TA1R1_105k = combined_data.loc[
        combined_data['dataset_original'].isin(
            ['Rocklin', 'Eva1', 'Eva2', 'Inna', 'Longxing', 'topology_mining_and_Longxing_chip_1',
             'topology_mining_and_Longxing_chip_2'])].copy()
    data_RD_BL_TA1R1_KJ_114k = combined_data.loc[
        combined_data['dataset_original'].isin(
            ['Rocklin', 'Eva1', 'Eva2', 'Inna', 'Longxing', 'topology_mining_and_Longxing_chip_1',
             'topology_mining_and_Longxing_chip_2', 'topology_mining_and_Longxing_chip_3'])].copy()

    # Grouping Data
    grouping_df = pd.read_csv(os.path.join(VERSIONED_DATA, 'protein-design/metadata/protein_groupings_by_uw.metadata.csv'),
                              comment='#', low_memory=False)
    grouping_df['dataset'] = grouping_df['dataset'].replace({"longxing_untested": "t_l_untested",
                                                             "topmining_untested": "t_l_untested"})
    print(grouping_df)

    feature_cols_to_normalize = ['AlaCount', 'T1_absq', 'T1_netq', 'Tend_absq', 'Tend_netq', 'Tminus1_absq',
                                 'Tminus1_netq', 'abego_res_profile', 'abego_res_profile_penalty',
                                 'avg_all_frags', 'avg_best_frag', 'bb', 'buns_bb_heavy', 'buns_nonheavy',
                                 'buns_sc_heavy', 'buried_minus_exposed', 'buried_np', 'buried_np_AFILMVWY',
                                 'buried_np_AFILMVWY_per_res', 'buried_np_per_res', 'buried_over_exposed',
                                 'chymo_cut_sites', 'chymo_with_LM_cut_sites', 'contact_all',
                                 'contact_core_SASA', 'contact_core_SCN', 'contig_not_hp_avg',
                                 'contig_not_hp_avg_norm', 'contig_not_hp_internal_max', 'contig_not_hp_max',
                                 'degree', 'dslf_fa13', 'entropy', 'exposed_hydrophobics',
                                 'exposed_np_AFILMVWY', 'exposed_polars', 'exposed_total', 'fa_atr',
                                 'fa_atr_per_res', 'fa_dun_dev', 'fa_dun_rot', 'fa_dun_semi', 'fa_elec',
                                 'fa_intra_atr_xover4', 'fa_intra_elec', 'fa_intra_rep_xover4',
                                 'fa_intra_sol_xover4', 'fa_rep', 'fa_rep_per_res', 'fa_sol', 'frac_helix',
                                 'frac_loop', 'frac_sheet', 'fxn_exposed_is_np', 'hbond_bb_sc', 'hbond_lr_bb',
                                 'hbond_lr_bb_per_sheet', 'hbond_sc', 'hbond_sr_bb', 'hbond_sr_bb_per_helix',
                                 'helix_sc', 'holes', 'hphob_sc_contacts', 'hphob_sc_degree', 'hxl_tors',
                                 'hydrophobicity', 'largest_hphob_cluster', 'lk_ball', 'lk_ball_bridge',
                                 'lk_ball_bridge_uncpl', 'lk_ball_iso', 'loop_sc', 'mismatch_probability',
                                 'n_charged', 'n_hphob_clusters', 'n_hydrophobic', 'n_hydrophobic_noA',
                                 'n_polar_core', 'n_res', 'nearest_chymo_cut_to_Cterm',
                                 'nearest_chymo_cut_to_Nterm', 'nearest_chymo_cut_to_term',
                                 'nearest_tryp_cut_to_Cterm', 'nearest_tryp_cut_to_Nterm',
                                 'nearest_tryp_cut_to_term', 'net_atr_net_sol_per_res', 'net_atr_per_res',
                                 'net_sol_per_res', 'netcharge', 'nres', 'nres_helix', 'nres_loop', 'nres_sheet',
                                 'omega', 'one_core_each', 'p_aa_pp', 'pack', 'percent_core_SASA',
                                 'percent_core_SCN', 'pro_close', 'rama_prepro', 'ref', 'res_count_core_SASA',
                                 'res_count_core_SCN', 'score_per_res', 'ss_contributes_core',
                                 'ss_sc', 'sum_best_frags', 'total_score', 'tryp_cut_sites', 'two_core_each',
                                 'worst6frags', 'worstfrag']

    '''
    train1, test1 = train_test_split(data_RD_16k, test_size=0.2, random_state=5,
                                     stratify=data_RD_16k[['topology', 'dataset_original']])
    train2, test2 = train1.copy(), combined_data.loc[
        combined_data['dataset_original'].isin(['Eva1', 'Eva2', 'Inna', 'Longxing'])].copy()
    train3, test3 = train_test_split(data_RD_BL_81k, test_size=0.2, random_state=5,
                                     stratify=data_RD_BL_81k[['topology', 'dataset_original']])
    train4, test4 = train3.copy(), combined_data.loc[combined_data['dataset_original'].isin(
        ['topology_mining_and_Longxing_chip_1', 'topology_mining_and_Longxing_chip_2'])].copy()
    train5, test5 = train_test_split(data_RD_BL_TA1R1_105k, test_size=0.2, random_state=5,
                                     stratify=data_RD_BL_TA1R1_105k[['topology', 'dataset_original']])
    train6, test6 = train5.copy(), combined_data.loc[
        combined_data['dataset_original'].isin(['topology_mining_and_Longxing_chip_3'])].copy()
    train7, test7 = train_test_split(data_RD_BL_TA1R1_KJ_114k, test_size=0.2, random_state=5,
                                     stratify=data_RD_BL_TA1R1_KJ_114k[['topology', 'dataset_original']])

    print()
    print(train1.shape, test1.shape, (train1.shape[0] + test1.shape[0]))
    print(train2.shape, test2.shape, (train2.shape[0] + test2.shape[0]))
    print(train3.shape, test3.shape, (train3.shape[0] + test3.shape[0]))
    print(train4.shape, test4.shape, (train4.shape[0] + test4.shape[0]))
    print(train5.shape, test5.shape, (train5.shape[0] + test5.shape[0]))
    print(train6.shape, test6.shape, (train6.shape[0] + test6.shape[0]))
    print(train7.shape, test7.shape, (train7.shape[0] + test7.shape[0]))
    print()

    # Change these:
    my_train = train1.copy()
    my_test = test1.copy()
    data_set_description = train_test_split_description = "1"
    col_to_predict = "stable?"

    # Regression:

    mr_linreg = linreg(my_train, my_test, col_to_predict, data_set_description, train_test_split_description)
    mr_rfr = rfr_features(my_train, my_test, col_to_predict, data_set_description, train_test_split_description)
    mr_seq = sequence_only_cnn(my_train, my_test, col_to_predict, data_set_description, train_test_split_description)

    perf_path = "general_results/performances_{}-{}-{}.csv".format(data_set_description, "linreg", col_to_predict)
    feat_path = "general_results/features_{}-{}-{}.csv".format(data_set_description, "linreg", col_to_predict)
    print("file name for performance results = {}".format(perf_path))
    print("file name for features results = {}".format(feat_path))
    th.run_model_general(mr_linreg, my_train, my_test, False, True, feature_cols_to_normalize, False, perf_path, feat_path)

    perf_path = "general_results/performances_{}-{}-{}.csv".format(data_set_description, "RFR", col_to_predict)
    feat_path = "general_results/features_{}-{}-{}.csv".format(data_set_description, "RFR", col_to_predict)
    print("file name for performance results = {}".format(perf_path))
    print("file name for features results = {}".format(feat_path))
    th.run_model_general(mr_rfr, my_train, my_test, False, True, feature_cols_to_normalize, True, perf_path, feat_path)

    perf_path = "general_results/performances_{}-{}-{}.csv".format(data_set_description, "CNN", col_to_predict)
    feat_path = "general_results/features_{}-{}-{}.csv".format(data_set_description, "CNN", col_to_predict)
    print("file name for performance results = {}".format(perf_path))
    print("file name for features results = {}".format(feat_path))
    th.run_model_general(mr_seq, my_train, my_test, True, False, None, False, perf_path, feat_path)


    # General Classification:
    mr_rfc = random_forest_classification(my_train, my_test, col_to_predict, data_set_description,
                                          train_test_split_description)
    logreg = logistic_classifier_topology_general_all_features(my_train, my_test, col_to_predict, data_set_description,
                                                               train_test_split_description)

    perf_path = "general_results/classification_performances_{}-{}-{}.csv".format(data_set_description, "logreg",
                                                                                  col_to_predict)
    feat_path = "general_results/classification_features_{}-{}-{}.csv".format(data_set_description, "logreg",
                                                                              col_to_predict)
    print("file name for performance results = {}".format(perf_path))
    print("file name for features results = {}".format(feat_path))
    th.run_model_general(logreg, my_train, my_test, False, True, feature_cols_to_normalize, False, perf_path,
                         feat_path)

    perf_path = "general_results/classification_performances_{}-{}-{}.csv".format(data_set_description, "RFC",
                                                                                  col_to_predict)
    feat_path = "general_results/classification_features_{}-{}-{}.csv".format(data_set_description, "RFC",
                                                                              col_to_predict)
    print("file name for performance results = {}".format(perf_path))
    print("file name for features results = {}".format(feat_path))
    th.run_model_general(mr_rfc, my_train, my_test, False, True, feature_cols_to_normalize, False, perf_path,
                         feat_path)
    '''

    # Leave one out Classification:
    # Change these values for different models/col_to_predict/data
    # model options: "RFC", "CNN", "logreg"
    # col_to_predict options: "stabilityscore_2classes", "stabilityscore_calibrated_2classes",
    #                           "stabilityscore_cnn_2classes", "stabilityscore_cnn_calibrated_2classes"
    # data_set_description options: "16k", "81k", "105k", "114k"
    # --------------
    model = "logreg"
    col_to_predict = "stabilityscore_2classes"
    data_set_description = "16k"
    # --------------

    if data_set_description == "16k":
        use_this_data = data_RD_16k
    elif data_set_description == "81k":
        use_this_data = data_RD_BL_81k
    elif data_set_description == "105k":
        use_this_data = data_RD_BL_TA1R1_105k
    elif data_set_description == "114k":
        use_this_data = data_RD_BL_TA1R1_KJ_114k
    else:
        raise ValueError("for this temporary analysis script, data_set_description must equal 16k, 81k, 105k, or 114k")

    perf_path = "leave_one_out_results/classification_performances_{}-{}-{}.csv".format(data_set_description, model,
                                                                                        col_to_predict)
    feat_path = "leave_one_out_results/classification_features_{}-{}-{}.csv".format(data_set_description, model,
                                                                                    col_to_predict)
    print("file name for performance results = {}".format(perf_path))
    print("file name for features results = {}".format(feat_path))
    print()

    if model == "RFC":
        th.run_model_on_grouping_splits(function_that_returns_model_runner=random_forest_classification,
                                        all_data_df=use_this_data, grouping_df=grouping_df,
                                        col_to_predict=col_to_predict, data_set_description=data_set_description,
                                        train_test_split_description="leave-one-group-out", normalize=True,
                                        feature_cols_to_normalize=feature_cols_to_normalize, get_pimportances=False,
                                        performance_output_path=perf_path, features_output_path=feat_path)
    # elif model == "CNN":
    #     th.run_model_on_grouping_splits(function_that_returns_model_runner=sequence_only_cnn,
    #                                     all_data_df=use_this_data, grouping_df=grouping_df,
    #                                     col_to_predict=col_to_predict, data_set_description=data_set_description,
    #                                     train_test_split_description="leave-one-group-out", normalize=False,
    #                                     feature_cols_to_normalize=None, get_pimportances=False,
    #                                     performance_output_path=perf_path, features_output_path=feat_path)
    elif model == "logreg":
        th.run_model_on_grouping_splits(function_that_returns_model_runner=logistic_classifier_topology_general_all_features,
                                        all_data_df=use_this_data, grouping_df=grouping_df,
                                        col_to_predict=col_to_predict, data_set_description=data_set_description,
                                        train_test_split_description="leave-one-group-out", normalize=True,
                                        feature_cols_to_normalize=feature_cols_to_normalize, get_pimportances=False,
                                        performance_output_path=perf_path, features_output_path=feat_path)
    else:
        raise ValueError("for this temporary analysis script, model must equal RFR, CNN, or Linreg")

    # Leave one group out regression runs, finished so commenting out for now to do general runs
    '''
    # Change these values for different models/col_to_predict/data
    # model options: "RFR", "CNN", "lingreg"
    # col_to_predict options: "stabilityscore", "stabilityscore_calibrated", "stabilityscore_cnn",
    #                         "stabilityscore_cnn_calibrated", "stabilityscore_calibrated_v2"
    # data_set_description options: "16k", "81k", "105k", "114k"
    # --------------
    model = "RFR"
    col_to_predict = "stabilityscore"
    data_set_description = "16k"
    # --------------

    if data_set_description == "16k":
        use_this_data = data_RD_16k
    elif data_set_description == "81k":
        use_this_data = data_RD_BL_81k
    elif data_set_description == "105k":
        use_this_data = data_RD_BL_TA1R1_105k
    elif data_set_description == "114k":
        use_this_data = data_RD_BL_TA1R1_KJ_114k
    else:
        raise ValueError("for this temporary analysis script, data_set_description must equal 16k, 81k, 105k, or 114k")

    perf_path = "leave_one_out_results/performances_{}-{}-{}.csv".format(data_set_description, model, col_to_predict)
    feat_path = "leave_one_out_results/features_{}-{}-{}.csv".format(data_set_description, model, col_to_predict)
    print("file name for performance results = {}".format(perf_path))
    print("file name for features results = {}".format(feat_path))
    print()

    if model == "RFR":
        th.run_model_on_grouping_splits(function_that_returns_model_runner=rfr_features,
                                        all_data_df=use_this_data, grouping_df=grouping_df,
                                        col_to_predict=col_to_predict, data_set_description=data_set_description,
                                        train_test_split_description="leave-one-group-out", normalize=True,
                                        feature_cols_to_normalize=feature_cols_to_normalize, get_pimportances=True,
                                        performance_output_path=perf_path, features_output_path=feat_path)
    elif model == "CNN":
        th.run_model_on_grouping_splits(function_that_returns_model_runner=sequence_only_cnn,
                                        all_data_df=use_this_data, grouping_df=grouping_df,
                                        col_to_predict=col_to_predict, data_set_description=data_set_description,
                                        train_test_split_description="leave-one-group-out", normalize=False,
                                        feature_cols_to_normalize=None, get_pimportances=False,
                                        performance_output_path=perf_path, features_output_path=feat_path)
    elif model == "linreg":
        th.run_model_on_grouping_splits(function_that_returns_model_runner=linreg,
                                        all_data_df=use_this_data, grouping_df=grouping_df,
                                        col_to_predict=col_to_predict, data_set_description=data_set_description,
                                        train_test_split_description="leave-one-group-out", normalize=True,
                                        feature_cols_to_normalize=feature_cols_to_normalize, get_pimportances=False,
                                        performance_output_path=perf_path, features_output_path=feat_path)
    else:
        raise ValueError("for this temporary analysis script, model must equal RFR, CNN, or Linreg")
    '''

    # th.run_test_harness()
    #
    # ccl_path = os.path.join(output_dir, 'comparable_classification_leaderboard.html')
    # crl_path = os.path.join(output_dir, 'comparable_regression_leaderboard.html')
    # gcl_path = os.path.join(output_dir, 'general_classification_leaderboard.html')
    # grl_path = os.path.join(output_dir, 'general_regression_leaderboard.html')
    #
    # # Build meta-leaderboard with DataTables styling
    # TEMPLATES = os.path.join(PARENT, 'templates')
    # index_path = os.path.join(output_dir, 'index.html')
    #
    # with open(index_path, 'w') as idx:
    #     with open(os.path.join(TEMPLATES, 'header.html.j2'), 'r') as hdr:
    #         for line in hdr:
    #             idx.write(line)
    #
    #     for lb in (ccl_path, crl_path, gcl_path, grl_path):
    #         fname = os.path.basename(lb)
    #         classname = fname.replace('_leaderboard.html', '')
    #         heading = classname.replace('_', ' ').title()
    #         idx.write('\n<h2>{}</h2>\n'.format(heading))
    #         with open(lb, 'r') as tbl:
    #             for line in tbl:
    #                 idx.write(line)
    #
    #     with open(os.path.join(TEMPLATES, 'footer.html.j2'), 'r') as ftr:
    #         for line in ftr:
    #             idx.write(line)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
