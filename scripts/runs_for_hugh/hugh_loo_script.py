import os
import argparse
import pandas as pd
from pathlib import Path
from test_harness.utils.names import Names
from test_harness.test_harness_class import TestHarness
from test_harness.data_wrangling import calculate_max_residues, encode_sequences
from test_harness.th_model_instances.hamed_models.rocklin_models import rocklins_logistic_classifier
from test_harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from test_harness.th_model_instances.jed_models.sequence_cnn_classification import sequence_only_cnn_classification

# At some point in your script you will need to define your data. For most cases the data will come from the `versioned_datasets` repo,
# which is why in this example script I am pointing to the data folder in the `versioned-datasets` repo:
# Ideally you would clone the `versioned-datasets` repo in the same location where you cloned the `protein-design` repo,
# but it shouldn't matter as long as you put the correct path here.
VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[4], 'versioned-datasets/data')
assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."
print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
print()

# this is where arguments are defined.
# argparse is useful for many situations such as using Launcher to run the same script with different arguments across multiple nodes
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=["Rocklin", "SD2-1", "SD2-2", "SD2-3"], required=True,
                    help="Which Dataset to use. Choices are: 'Rocklin', 'SD2-1', 'SD2-2'', 'SD2-3'")
parser.add_argument('--col_to_pred', required=True, help="Name of column to predict.")


def main(args):
    # Reading in data from versioned-datasets repo.
    # Using the versioned-datasets repo is probably what most people want to do, but you can read in your data however you like.
    combined_data = pd.read_csv(os.path.join(VERSIONED_DATA, 'protein-design/aggregated_data/all_libs_cleaned.v3.aggregated_data.csv'),
                                comment='#', low_memory=False)

    # The following lines of code are just modifications to the dataframe that was read in (pre-processing).
    combined_data['dataset_original'] = combined_data['dataset']
    combined_data['dataset'] = combined_data['dataset'].replace({"topology_mining_and_Longxing_chip_1": "t_l_untested",
                                                                 "topology_mining_and_Longxing_chip_2": "t_l_untested",
                                                                 "topology_mining_and_Longxing_chip_3": "t_l_untested"})
    col_order = list(combined_data.columns.values)
    col_order.insert(2, col_order.pop(col_order.index('dataset_original')))
    combined_data = combined_data[col_order]
    combined_data['stabilityscore_2classes'] = combined_data['stabilityscore'] > 1
    combined_data['stabilityscore_calibrated_2classes'] = combined_data['stabilityscore_calibrated'] > 1
    combined_data['stabilityscore_cnn_2classes'] = combined_data['stabilityscore_cnn'] > 1
    combined_data['stabilityscore_cnn_calibrated_2classes'] = combined_data['stabilityscore_cnn_calibrated'] > 1

    # Grouping Dataframe read in for leave-one-out analysis.
    grouping_df = pd.read_csv(os.path.join(VERSIONED_DATA, 'protein-design/metadata/protein_groupings_by_uw.v1.metadata.csv'), comment='#',
                              low_memory=False)
    grouping_df['dataset'] = grouping_df['dataset'].replace({"longxing_untested": "t_l_untested",
                                                             "topmining_untested": "t_l_untested"})

    # list of feature columns to normalize:
    feature_cols = ['AlaCount', 'T1_absq', 'T1_netq', 'Tend_absq', 'Tend_netq', 'Tminus1_absq',
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

    # Here I am creating different data subsets based on the different rounds of protein design data we had on SD2
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

    # Here I set the path to the directory in which I want my results to go by setting the output_location argument. When the Test Harness
    # runs, it will create a "test_harness_results" folder inside of output_location and place all results inside the "test_harness_results"
    # folder. If the "test_harness_results" folder already exists, then previous results/leaderboards will be updated.
    # In this example script, the output_location has been set to the "examples" folder to separate example results from other ones.
    current_folder_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(current_folder_path))
    print()
    th = TestHarness(output_location=current_folder_path)

    dataset_dict = {"Rocklin": data_RD_16k, "SD2-1": data_RD_BL_81k, "SD2-2": data_RD_BL_TA1R1_105k, "SD2-3": data_RD_BL_TA1R1_KJ_114k}

    colpred = args.col_to_pred
    dataset_key = args.dataset
    dataset = dataset_dict[dataset_key].copy()

    th.run_leave_one_out(function_that_returns_TH_model=rocklins_logistic_classifier, dict_of_function_parameters={},
                         data=dataset, data_description=dataset_key, grouping=grouping_df,
                         grouping_description="protein_groupings_by_uw.v1", cols_to_predict=colpred, feature_cols_to_use=feature_cols,
                         normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=False)

    th.run_leave_one_out(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={},
                         data=dataset, data_description=dataset_key, grouping=grouping_df,
                         grouping_description="protein_groupings_by_uw.v1", cols_to_predict=colpred, feature_cols_to_use=feature_cols,
                         normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=Names.RFPIMP_PERMUTATION)

    # max_residues = calculate_max_residues([dataset])
    # dataset_encoded = encode_sequences(dataset, max_residues)
    # th.run_leave_one_out(function_that_returns_TH_model=sequence_only_cnn_classification,
    #                      dict_of_function_parameters={"max_residues": max_residues, "padding": 14, "assign_class_weights": False},
    #                      data=dataset_encoded, data_description=dataset_key, grouping=grouping_df,
    #                      grouping_description="protein_groupings_by_uw.v1", cols_to_predict=colpred,
    #                      feature_cols_to_use=["encoded_sequence"], normalize=False, feature_cols_to_normalize=None,
    #                      feature_extraction=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
