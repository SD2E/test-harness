import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.utils.names import Names
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression

from harness.th_model_instances.hamed_models.keras_classification import keras_classification_4


VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."
print()


def main():
    # Reading in data from versioned-datasets repo.
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
    combined_data['stabilityscore_cnn_calibrated_2classes'] = combined_data['stabilityscore_cnn_calibrated'] > 1

    # Using a subset of the data for testing, and making custom train/test splits.
    data_RD_16k = combined_data.loc[combined_data['dataset_original'] == 'Rocklin'].copy()
    train, test = train_test_split(data_RD_16k, test_size=0.2, random_state=5, stratify=data_RD_16k[['topology', 'dataset_original']])
    toy_train = train.sample(n=100, random_state=5)
    toy_test = test.sample(n=100, random_state=5)

    # Grouping Dataframe read in for leave-one-out analysis.
    grouping_df = pd.read_csv(os.path.join(VERSIONED_DATA, 'protein-design/metadata/protein_groupings_by_uw.v1.metadata.csv'), comment='#',
                              low_memory=False)
    grouping_df['dataset'] = grouping_df['dataset'].replace({"longxing_untested": "t_l_untested",
                                                             "topmining_untested": "t_l_untested"})

    # list of feature columns to use and/or normalize:
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

    toy_feature_cols = feature_cols[:10]

    # TestHarness usage starts here, all code before this was just data input and pre-processing.
    current_path = "/Users/janestrada/Desktop/SorelleSummerResearch/TestHarness/"
    print("initializing TestHarness object with output_location equal to {}".format(current_path))
    print()
    th = TestHarness(output_location=current_path)

    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=toy_train,
                  testing_data=toy_test, data_and_split_description="testing shap on toy datasets",
                  cols_to_predict='stabilityscore_cnn_calibrated_2classes',
                  feature_cols_to_use=toy_feature_cols, normalize=True, feature_cols_to_normalize=toy_feature_cols,
                  feature_extraction='bba_audit', predict_untested_data=False,
                  interpret_complex_model=True)

    th.run_custom(function_that_returns_TH_model=keras_classification_4, dict_of_function_parameters={}, training_data=toy_train,
                  testing_data=toy_test, data_and_split_description="testing shap on toy datasets",
                  cols_to_predict='stabilityscore_cnn_calibrated_2classes',
                  feature_cols_to_use=toy_feature_cols, normalize=True, feature_cols_to_normalize=toy_feature_cols,
                  feature_extraction='bba_audit', predict_untested_data=False,
                  interpret_complex_model=True)
    
if __name__ == '__main__':
    main()
