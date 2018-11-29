#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import datetime as dt
import os
import importlib
from pathlib import Path

import types
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from test_harness.test_harness_class import TestHarness

from test_harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification


# In[2]:



PWD = os.getcwd()
# HERE = Path().resolve()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)
RESULTSPATH = os.path.dirname(PARENT)
print("PWD:", PWD)
print("HERE:", HERE)
print("PARENT:", PARENT)
print("RESULTSPATH:", RESULTSPATH)
print()
# SET PATH TO DATA FOLDER IN LOCALLY CLONED `versioned-datasets` REPO HERE:
# Note that if you clone the `versioned-datasets` repo at the same level as where you cloned the `protein-design` repo,
# then you can use VERSIONED_DATASETS = os.path.join(HERE.parents[2], 'versioned-datasets/data')
# VERSIONED_REPO_PATH = os.path.join(Path(HERE).resolve().parents[1], 'versioned-datasets/data')
VERSIONED_REPO_PATH = os.path.join(Path(__file__).resolve().parents[3], 'versioned-datasets/data')

print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_REPO_PATH))
print()


# In[17]:


included_chips = {'topology_mining_and_Longxing_chip_1',
                  'topology_mining_and_Longxing_chip_2',
                  'topology_mining_and_Longxing_chip_3'}

# load dfs with structural info from rosetta about the proteins in the stability experiment
chip1_structure_DF = pd.read_csv(
    VERSIONED_REPO_PATH + '/protein-design/structural_metrics/topology_mining_and_Longxing_chip_1.v1.structural_metrics.csv',
    comment='#')
chip2_structure_DF = pd.read_csv(
    VERSIONED_REPO_PATH + '/protein-design/structural_metrics/topology_mining_and_Longxing_chip_2.v1.structural_metrics.csv',
    comment='#')
chip3_structure_DF = pd.read_csv(
    VERSIONED_REPO_PATH + '/protein-design/structural_metrics/topology_mining_and_Longxing_chip_3.v1.structural_metrics.csv',
    comment='#')
structure_df_full = pd.concat([chip1_structure_DF, chip2_structure_DF, chip3_structure_DF], axis=0)

# these have some string 'Runtime Errors' that we would have to handle as NaNs.  We might want to do that
structure_df_full.drop(['ss_sc', 'loop_sc'], axis=1, inplace=True)
structure_df_full.topology.fillna('unknown', inplace=True)

structure_df_full = structure_df_full.set_index(['dataset', 'name'])


# In[29]:


low_thru_chip1_df = pd.read_csv(
    VERSIONED_REPO_PATH + '/protein-design/experimental_stability_scores/topology_mining_and_Longxing_chip_1.v6.experimental_stability_scores.csv',
    comment='#')
low_thru_chip2_df = pd.read_csv(
    VERSIONED_REPO_PATH + '/protein-design/experimental_stability_scores/topology_mining_and_Longxing_chip_2.v6.experimental_stability_scores.csv',
    comment='#')
low_thru_chip3_df = pd.read_csv(
    VERSIONED_REPO_PATH + '/protein-design/experimental_stability_scores/topology_mining_and_Longxing_chip_3.v6.experimental_stability_scores.csv',
    comment='#')
low_thru_df_combined = pd.concat([low_thru_chip1_df, low_thru_chip2_df, low_thru_chip3_df])
low_thru_df_combined = low_thru_df_combined.set_index(['dataset', 'name'], drop=False)
low_thru_data = low_thru_df_combined.join(structure_df_full)


# In[30]:


high_thru_data = pd.read_csv(os.path.join(VERSIONED_REPO_PATH, 'protein-design/experimental_stability_scores/topology_mining_and_Longxing_all_chips.experimental_stability_scores.csv'),
                             comment='#', low_memory=False)
high_thru_data = high_thru_data.set_index(['dataset', 'name'], drop=False)
# import ipdb; ipdb.set_trace()

high_thru_data = high_thru_data.join(structure_df_full, rsuffix='_structure_df_column')


# In[32]:


rows = low_thru_data.shape[0]
low_thru_data.dropna(inplace=True)
dropped_rows = rows - low_thru_data.shape[0]
print("Dropped %s rows from low thru data" % dropped_rows)

rows = high_thru_data.shape[0]
high_thru_data.dropna(inplace=True)
dropped_rows - high_thru_data.shape[0]
print("Dropped %s rows from high thru data" % dropped_rows)


# In[33]:


dfs = [low_thru_data, high_thru_data]

for df in dfs:
    df['dataset_original'] = df['dataset']
    df['dataset'] = df['dataset'].replace(
        {"topology_mining_and_Longxing_chip_1": "t_l_untested",
         "topology_mining_and_Longxing_chip_2": "t_l_untested",
         "topology_mining_and_Longxing_chip_3": "t_l_untested"})
    # col_order = list(df.columns.values)
    # col_order.insert(2, col_order.pop(col_order.index('dataset_original')))
    # df = df[col_order]
    df['stabilityscore_2classes'] = df['stabilityscore'] > 1

    # todo: why aren't these in the high-thru dataset?
    # df['stabilityscore_calibrated_2classes'] = df['stabilityscore_calibrated'] > 1
    # df['stabilityscore_cnn_2classes'] = df['stabilityscore_cnn'] > 1
    # df['stabilityscore_cnn_calibrated_2classes'] = df['stabilityscore_cnn_calibrated'] > 1


# In[ ]:





# In[35]:


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
                             'lk_ball_bridge_uncpl', 'lk_ball_iso', 'mismatch_probability',
                             'n_charged', 'n_hphob_clusters', 'n_hydrophobic', 'n_hydrophobic_noA',
                             'n_polar_core', 'n_res', 'nearest_chymo_cut_to_Cterm',
                             'nearest_chymo_cut_to_Nterm', 'nearest_chymo_cut_to_term',
                             'nearest_tryp_cut_to_Cterm', 'nearest_tryp_cut_to_Nterm',
                             'nearest_tryp_cut_to_term', 'net_atr_net_sol_per_res', 'net_atr_per_res',
                             'net_sol_per_res', 'netcharge', 'nres', 'nres_helix', 'nres_loop', 'nres_sheet',
                             'omega', 'one_core_each', 'p_aa_pp', 'pack', 'percent_core_SASA',
                             'percent_core_SCN', 'pro_close', 'rama_prepro', 'ref', 'res_count_core_SASA',
                             'res_count_core_SCN', 'score_per_res', 'ss_contributes_core',
                              'sum_best_frags', 'total_score', 'tryp_cut_sites', 'two_core_each',
                             'worst6frags', 'worstfrag',
                             #'ss_sc', 'loop_sc', # removed because of some NaNs
                             ]


# In[36]:


# TestHarness usage starts here, all code before this was just data input code.
th = TestHarness(output_path=RESULTSPATH)
rf_classification_model = random_forest_classification()

th.add_custom_runs(test_harness_models=rf_classification_model, training_data=high_thru_data, testing_data=low_thru_data,
                   data_and_split_description="train on low thru, test on high thru",
                   cols_to_predict=['stabilityscore_2classes'],
                   feature_cols_to_use=feature_cols_to_normalize, normalize=True, feature_cols_to_normalize=feature_cols_to_normalize,
                   feature_extraction=False, predict_untested_data=False)

# Grouping Dataframe read in for leave-one-out analysis.
grouping_df = pd.read_csv(os.path.join(VERSIONED_REPO_PATH,
                                       'protein-design/metadata/protein_groupings_by_uw.metadata.csv'), comment='#',
                          low_memory=False)
grouping_df['dataset'] = grouping_df['dataset'].replace({"longxing_untested": "t_l_untested",
                                                         "topmining_untested": "t_l_untested"})


rf_classification_model = random_forest_classification()
th.add_leave_one_out_runs(test_harness_models=rf_classification_model, data=low_thru_data, data_description="low_thru_data",
                          grouping=grouping_df, grouping_description="grouping_df", cols_to_predict='stabilityscore_2classes',
                          feature_cols_to_use=feature_cols_to_normalize, normalize=True,
                          feature_cols_to_normalize=feature_cols_to_normalize, feature_extraction=False)



th.execute_runs()





