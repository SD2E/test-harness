import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification

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
parser.add_argument('--data_split_number', required=True, help='Enter number between 1 and 7')


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

    # Here I am creating 7 train/test splits based on the different data rounds defined above
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

    # I use this dictionary to be able to pass the numbers 1-7 as argparse arguments for my script
    train_test_dict = {"1": (train1, test1), "2": (train2, test2), "3": (train3, test3), "4": (train4, test4), "5": (train5, test5),
                       "6": (train6, test6), "7": (train7, test7)}

    # Retrieve data split argument from argparse and create train_split and test_split, which are used by the test harness runs
    data_split_number = args.data_split_number
    train_split = train_test_dict[data_split_number][0].copy()
    test_split = train_test_dict[data_split_number][1].copy()

    # Here I set the path to the directory in which I want my results to go by setting the output_location argument. When the Test Harness
    # runs, it will create a "test_harness_results" folder inside of output_location and place all results inside the "test_harness_results"
    # folder. If the "test_harness_results" folder already exists, then previous results/leaderboards will be updated.
    # In this example script, the output_location has been set to the "examples" folder to separate example results from other ones.
    examples_folder_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(examples_folder_path))
    print()
    th = TestHarness(output_location=examples_folder_path)

    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train_split,
                  testing_data=test_split,
                  data_and_split_description="example argparse custom run on Rocklin data, split = {}".format(data_split_number),
                  cols_to_predict='stabilityscore_2classes',
                  feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                  feature_extraction=False, predict_untested_data=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
