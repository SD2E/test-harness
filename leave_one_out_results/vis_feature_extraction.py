import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
from random import randint
from collections import Counter
import matplotlib.pyplot as plt
from feature_importance.graphs import Graph

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    files = [f for f in files if '.csv' in f]
    feature_files = [f for f in files if 'features' in f]

    # pimportance_df = None
    # for f in feature_files:
    #     print(f)
    #     dataset = f.split("_")[1].split("-")[0]
    #     model_used = f.split("_")[1].split("-")[1]
    #     col_predicted = f.split("-")[2].split(".csv")[0]
    #     print(dataset)
    #     print(model_used)
    #     print(col_predicted)
    #     df = pd.read_csv(f, comment="#")
    #     print(df.head())
    #     print()
    #     df.rename(columns={"Importance": "{}_Dataset_{}".format(model_used, dataset)}, inplace=True)
    #     if pimportance_df is None:
    #         pimportance_df = df.copy()
    #     else:
    #         pimportance_df = pd.merge(pimportance_df, df, on="Feature")
    # # print(pimportance_df)
    #
    # pimportance_cols = list(pimportance_df.columns.values)
    # pimportance_cols.remove("Feature")
    #
    # percent_overlap_heatmap(pimportance_df, pimportance_cols, 10)





    features_114k_RFR_stabilityscore = pd.read_csv("features_114k-RFR-stabilityscore.csv")
    features_114k_RFR_stabilityscore['col_predicted'] = "stabilityscore"
    features_114k_RFR_stabilityscore_calibrated = pd.read_csv("features_114k-RFR-stabilityscore_calibrated.csv")
    features_114k_RFR_stabilityscore_calibrated['col_predicted'] = "stabilityscore_calibrated"
    features_114k_RFR_stabilityscore_calibrated_v2 = pd.read_csv("features_114k-RFR-stabilityscore_calibrated_v2.csv")
    features_114k_RFR_stabilityscore_calibrated_v2['col_predicted'] = "stabilityscore_calibrated_v2"
    features_114k_RFR_stabilityscore_cnn = pd.read_csv("features_114k-RFR-stabilityscore_cnn.csv")
    features_114k_RFR_stabilityscore_cnn['col_predicted'] = "stabilityscore_USM"
    features_114k_RFR_stabilityscore_cnn_calibrated = pd.read_csv("features_114k-RFR-stabilityscore_cnn_calibrated.csv")
    features_114k_RFR_stabilityscore_cnn_calibrated['col_predicted'] = "stabilityscore_USM_calibrated"

    n = 10

    # features_114k_RFR_stabilityscore
    group_cols = list(features_114k_RFR_stabilityscore.columns.values)
    group_cols.remove("Feature")
    group_cols.remove("col_predicted")
    feature_counts_df = pd.DataFrame()
    feature_counts_df["feature"] = list(features_114k_RFR_stabilityscore['Feature'])
    feature_counts_df["number_of_groups_it_is_present_in"] = 0
    feature_counts_df["groups_it_is_present_in"] = None
    feature_counts_df["col_predicted"] = "stabilityscore"
    for group_col in group_cols:
        sorted_df = features_114k_RFR_stabilityscore[["Feature", group_col]].copy()
        sorted_df = sorted_df.sort_values(by=group_col, ascending=False)
        top_n = list(sorted_df["Feature"][:n])
        for f in top_n:
            feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"] = \
                feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"].item() + 1
            current_groups = feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"].item()
            if current_groups is None:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = group_col
            else:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = \
                    current_groups + '; ' + group_col
    feature_counts_df1 = feature_counts_df.sort_values(by="number_of_groups_it_is_present_in", ascending=False)

    
    
    # features_114k_RFR_stabilityscore_calibrated
    group_cols = list(features_114k_RFR_stabilityscore_calibrated.columns.values)
    group_cols.remove("Feature")
    group_cols.remove("col_predicted")
    feature_counts_df = pd.DataFrame()
    feature_counts_df["feature"] = list(features_114k_RFR_stabilityscore_calibrated['Feature'])
    feature_counts_df["number_of_groups_it_is_present_in"] = 0
    feature_counts_df["groups_it_is_present_in"] = None
    feature_counts_df["col_predicted"] = "stabilityscore_calibrated"
    for group_col in group_cols:
        sorted_df = features_114k_RFR_stabilityscore_calibrated[["Feature", group_col]].copy()
        sorted_df = sorted_df.sort_values(by=group_col, ascending=False)
        top_n = list(sorted_df["Feature"][:n])
        for f in top_n:
            feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"] = \
                feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"].item() + 1
            current_groups = feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"].item()
            if current_groups is None:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = group_col
            else:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = \
                    current_groups + '; ' + group_col
    feature_counts_df2 = feature_counts_df.sort_values(by="number_of_groups_it_is_present_in", ascending=False)

    
    
    # features_114k_RFR_stabilityscore_calibrated_v2
    group_cols = list(features_114k_RFR_stabilityscore_calibrated_v2.columns.values)
    group_cols.remove("Feature")
    group_cols.remove("col_predicted")
    feature_counts_df = pd.DataFrame()
    feature_counts_df["feature"] = list(features_114k_RFR_stabilityscore_calibrated_v2['Feature'])
    feature_counts_df["number_of_groups_it_is_present_in"] = 0
    feature_counts_df["groups_it_is_present_in"] = None
    feature_counts_df["col_predicted"] = "stabilityscore_calibrated_v2"
    for group_col in group_cols:
        sorted_df = features_114k_RFR_stabilityscore_calibrated_v2[["Feature", group_col]].copy()
        sorted_df = sorted_df.sort_values(by=group_col, ascending=False)
        top_n = list(sorted_df["Feature"][:n])
        for f in top_n:
            feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"] = \
                feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"].item() + 1
            current_groups = feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"].item()
            if current_groups is None:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = group_col
            else:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = \
                    current_groups + '; ' + group_col
    feature_counts_df3 = feature_counts_df.sort_values(by="number_of_groups_it_is_present_in", ascending=False)

    
    
    # features_114k_RFR_stabilityscore_cnn
    group_cols = list(features_114k_RFR_stabilityscore_cnn.columns.values)
    group_cols.remove("Feature")
    group_cols.remove("col_predicted")
    feature_counts_df = pd.DataFrame()
    feature_counts_df["feature"] = list(features_114k_RFR_stabilityscore_cnn['Feature'])
    feature_counts_df["number_of_groups_it_is_present_in"] = 0
    feature_counts_df["groups_it_is_present_in"] = None
    feature_counts_df["col_predicted"] = "stabilityscore_USM"
    for group_col in group_cols:
        sorted_df = features_114k_RFR_stabilityscore_cnn[["Feature", group_col]].copy()
        sorted_df = sorted_df.sort_values(by=group_col, ascending=False)
        top_n = list(sorted_df["Feature"][:n])
        for f in top_n:
            feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"] = \
                feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"].item() + 1
            current_groups = feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"].item()
            if current_groups is None:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = group_col
            else:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = \
                    current_groups + '; ' + group_col
    feature_counts_df4 = feature_counts_df.sort_values(by="number_of_groups_it_is_present_in", ascending=False)

    
    
    # features_114k_RFR_stabilityscore_cnn_calibrated
    group_cols = list(features_114k_RFR_stabilityscore_cnn_calibrated.columns.values)
    group_cols.remove("Feature")
    group_cols.remove("col_predicted")
    feature_counts_df = pd.DataFrame()
    feature_counts_df["feature"] = list(features_114k_RFR_stabilityscore_cnn_calibrated['Feature'])
    feature_counts_df["number_of_groups_it_is_present_in"] = 0
    feature_counts_df["groups_it_is_present_in"] = None
    feature_counts_df["col_predicted"] = "stabilityscore_USM_calibrated"
    for group_col in group_cols:
        sorted_df = features_114k_RFR_stabilityscore_cnn_calibrated[["Feature", group_col]].copy()
        sorted_df = sorted_df.sort_values(by=group_col, ascending=False)
        top_n = list(sorted_df["Feature"][:n])
        for f in top_n:
            feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"] = \
                feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"].item() + 1
            current_groups = feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"].item()
            if current_groups is None:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = group_col
            else:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = \
                    current_groups + '; ' + group_col
    feature_counts_df5 = feature_counts_df.sort_values(by="number_of_groups_it_is_present_in", ascending=False)


    frames = [feature_counts_df1, feature_counts_df2, feature_counts_df3, feature_counts_df4, feature_counts_df5]

    combod = pd.concat(frames)
    print(combod)
    # combod = combod.loc[combod['number_of_groups_it_is_present_in'] > 14]

    # print(feature_counts_df1.loc[feature_counts_df1["feature"]=="avg_all_frags"])
    # print(feature_counts_df2.loc[feature_counts_df2["feature"]=="avg_all_frags"])
    # print(feature_counts_df3.loc[feature_counts_df3["feature"]=="avg_all_frags"])
    # print(feature_counts_df4.loc[feature_counts_df4["feature"]=="avg_all_frags"])
    # print(feature_counts_df5.loc[feature_counts_df5["feature"]=="avg_all_frags"])
    # print(combod)

    combod_plot = combod.loc[combod["feature"].isin(["hydrophobicity", "buried_np_AFILMVWY_per_res",
                                                     "abego_res_profile_penalty", "avg_all_frags",
                                                     "exposed_np_AFILMVWY", "mismatch_probability"])]

    ax = sns.barplot(x="feature", y="number_of_groups_it_is_present_in", data=combod_plot, hue="col_predicted")
    ax.set_xlabel("Rosetta Feature", fontsize=15)
    ax.set_ylabel("Number of Times Feature was Extracted Per Topology", fontsize=15)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
    ax.legend(loc='upper right')
    plt.gcf().subplots_adjust(bottom=0.25)

    # plt.tight_layout()
    plt.show()

    
    
    
    
    
    
    
    
    
    
    import sys
    sys.exit(0)









    n = 10
    num_bars = 40

    # features_114k_RFR_stabilityscore
    group_cols = list(features_114k_RFR_stabilityscore.columns.values)
    group_cols.remove("Feature")
    features_114k_RFR_stabilityscore['mean'] = features_114k_RFR_stabilityscore[group_cols].mean(axis=1)
    features_114k_RFR_stabilityscore['max'] = features_114k_RFR_stabilityscore[group_cols].max(axis=1)
    features_114k_RFR_stabilityscore['idx_max'] = features_114k_RFR_stabilityscore[group_cols].idxmax(axis=1)
    features_114k_RFR_stabilityscore.sort_values(by="mean", axis=0, inplace=True, ascending=False)
    print(features_114k_RFR_stabilityscore.head(15))
    print()
    features_114k_RFR_stabilityscore.sort_values(by="max", axis=0, inplace=True, ascending=False)
    print(features_114k_RFR_stabilityscore.head(15))
    print()
    feature_counts_df = pd.DataFrame()
    feature_counts_df["feature"] = list(features_114k_RFR_stabilityscore['Feature'])
    feature_counts_df["number_of_groups_it_is_present_in"] = 0
    feature_counts_df["groups_it_is_present_in"] = None
    for group_col in group_cols:
        sorted_df = features_114k_RFR_stabilityscore[["Feature", group_col]].copy()
        sorted_df = sorted_df.sort_values(by=group_col, ascending=False)
        top_n = list(sorted_df["Feature"][:n])
        for f in top_n:
            feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"] = \
                feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"].item() + 1
            current_groups = feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"].item()
            if current_groups is None:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = group_col
            else:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = \
                    current_groups + '; ' + group_col
    feature_counts_df = feature_counts_df.sort_values(by="number_of_groups_it_is_present_in", ascending=False)
    print(feature_counts_df.head(55))
    ax = sns.barplot(x="number_of_groups_it_is_present_in", y="feature", data=feature_counts_df[:num_bars])
    ax.set_ylabel("Rosetta Feature", fontsize=15)
    ax.set_xlabel("Number of Groups/Topologies in which Feature is in Top 10 Most Important Features", fontsize=15)
    plt.tight_layout()
    plt.show()


    # features_114k_RFR_stabilityscore_calibrated
    group_cols = list(features_114k_RFR_stabilityscore_calibrated.columns.values)
    group_cols.remove("Feature")
    features_114k_RFR_stabilityscore_calibrated['mean'] = features_114k_RFR_stabilityscore_calibrated[group_cols].mean(axis=1)
    features_114k_RFR_stabilityscore_calibrated['max'] = features_114k_RFR_stabilityscore_calibrated[group_cols].max(axis=1)
    features_114k_RFR_stabilityscore_calibrated['idx_max'] = features_114k_RFR_stabilityscore_calibrated[group_cols].idxmax(axis=1)
    features_114k_RFR_stabilityscore_calibrated.sort_values(by="mean", axis=0, inplace=True, ascending=False)
    print(features_114k_RFR_stabilityscore_calibrated.head(15))
    print()
    features_114k_RFR_stabilityscore_calibrated.sort_values(by="max", axis=0, inplace=True, ascending=False)
    print(features_114k_RFR_stabilityscore_calibrated.head(15))
    print()
    feature_counts_df = pd.DataFrame()
    feature_counts_df["feature"] = list(features_114k_RFR_stabilityscore_calibrated['Feature'])
    feature_counts_df["number_of_groups_it_is_present_in"] = 0
    feature_counts_df["groups_it_is_present_in"] = None
    for group_col in group_cols:
        sorted_df = features_114k_RFR_stabilityscore_calibrated[["Feature", group_col]].copy()
        sorted_df = sorted_df.sort_values(by=group_col, ascending=False)
        top_n = list(sorted_df["Feature"][:n])
        for f in top_n:
            feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"] = \
                feature_counts_df.loc[feature_counts_df["feature"] == f, "number_of_groups_it_is_present_in"].item() + 1
            current_groups = feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"].item()
            if current_groups is None:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = group_col
            else:
                feature_counts_df.loc[feature_counts_df["feature"] == f, "groups_it_is_present_in"] = \
                    current_groups + '; ' + group_col
    feature_counts_df = feature_counts_df.sort_values(by="number_of_groups_it_is_present_in", ascending=False)
    print(feature_counts_df.head(55))
    ax = sns.barplot(x="number_of_groups_it_is_present_in", y="feature", data=feature_counts_df[:num_bars])
    plt.tight_layout()
    plt.show()







'''
# percent_overlap_heatmap(features_114k_RFR_stabilityscore, cols, 10)

# feature_mappings = {"cavity_volume": "rosetta_filters", "degree": "rosetta_filters",
#                     "contact_all": "rosetta_filters",
#                     "exposed_hydrophobics": "rosetta_filters", "exposed_polars": "rosetta_filters",
#                     "exposed_total": "rosetta_filters", "fxn_exposed_is_np": "rosetta_filters",
#                     "holes": "rosetta_filters",
#                     "helix_sc": "rosetta_filters", "loop_sc": "rosetta_filters",
#                     "mismatch_probability": "rosetta_filters",
#                     "pack": "rosetta_filters", "unsat_hbond": "rosetta_filters", "ss_sc": "rosetta_filters",
#                     "unsat_hbond2": "rosetta_filters",
#                     "score_per_res": "energy_terms", "fa_atr_per_res": "energy_terms",
#                     "fa_rep_per_res": "energy_terms",
#                     "hbond_lr_bb_per_res": "energy_terms", "hbond_lr_bb_per_sheet": "energy_terms",
#                     "hbond_sr_bb_per_helix": "energy_terms", "net_atr_per_res": "energy_terms",
#                     "net_sol_per_res": "energy_terms", "net_atr_net_sol_per_res": "energy_terms",
#                     "dslf_fa13": "energy_terms",
#                     "fa_atr": "energy_terms", "fa_dun": "energy_terms", "fa_elec": "energy_terms",
#                     "fa_intra_rep": "energy_terms", "fa_intra_sol_xover4": "energy_terms", "fa_rep": "energy_terms",
#                     "fa_sol": "energy_terms", "hbond_bb_sc": "energy_terms", "hbond_lr_bb": "energy_terms",
#                     "hbond_sc": "energy_terms", "hbond_sr_bb": "energy_terms", "lk_ball_wtd": "energy_terms",
#                     "omega": "energy_terms", "p_aa_pp": "energy_terms", "pro_close": "energy_terms",
#                     "rama_prepro": "energy_terms", "ref": "energy_terms", "ss_sc": "energy_terms",
#                     "total_score": "energy_terms", "yhh_planarity": "energy_terms", "lk_ball": "energy_terms",
#                     "lk_ball_bridge": "energy_terms", "lk_ball_bridge_uncpl": "energy_terms",
#                     "lk_ball_iso": "energy_terms",
#                     "description": "seq_top_feats", "sequence": "seq_top_feats", "dssp": "seq_top_feats",
#                     "nres": "seq_top_feats", "n_res": "seq_top_feats", "nres_helix": "seq_top_feats",
#                     "nres_sheet": "seq_top_feats", "nres_loop": "seq_top_feats", "frac_helix": "seq_top_feats",
#                     "frac_sheet": "seq_top_feats", "frac_loop": "seq_top_feats", "n_charged": "seq_top_feats",
#                     "netcharge": "seq_top_feats", "AlaCount": "seq_top_feats", "n_hydrophobic": "seq_top_feats",
#                     "n_hydrophobic_noA": "seq_top_feats",
#                     "buried_np": "custom_rosetta", "buried_np_per_res": "custom_rosetta",
#                     "buried_minus_exposed": "custom_rosetta", "buried_np_afilmvwy": "custom_rosetta",
#                     "buried_np_afilmvwy_per_res": "custom_rosetta", "buried_over_exposed": "custom_rosetta",
#                     "exposed_np_afilmvwy": "custom_rosetta", "one_core_each": "custom_rosetta",
#                     "two_core_each": "custom_rosetta", "ss_contributes_core": "custom_rosetta",
#                     "res_count_core_SASA": "custom_rosetta", "res_count_core_SCN": "custom_rosetta",
#                     "percent_core_SASA": "custom_rosetta", "percent_core_SCN": "custom_rosetta",
#                     "abego_res_profile": "abego", "abego_res_profile_penalty": "abego",
#                     "contig_not_hp_avg": "hydrophobicity", "contig_not_hp_avg_norm": "hydrophobicity",
#                     "contig_not_hp_norm": "hydrophobicity", "contig_not_hp_max": "hydrophobicity",
#                     "contig_not_hp_internal_max": "hydrophobicity", "hphob_sc_contacts": "hydrophobicity",
#                     "hphob_sc_degree": "hydrophobicity", "largest_hphob_cluster": "hydrophobicity",
#                     "n_hphob_clusters": "hydrophobicity", "hydrophobicity": "hydrophobicity",
#                     "chymo_cut_sites": "trypsin_cut", "chymo_with_LM_cut_sites": "trypsin_cut",
#                     "tryp_cut_sites": "trypsin_cut",
#                     "nearest_chymo_cut_to_Nterm": "trypsin_cut", "nearest_chymo_cut_to_term": "trypsin_cut",
#                     "nearest_tryp_cut_to_term": "trypsin_cut", "nearest_chymo_cut_to_cterm": "trypsin_cut",
#                     "nearest_tryp_cut_to_cterm": "trypsin_cut", "nearest_tryp_cut_to_nterm": "trypsin_cut",
#                     "T1_netq": "helix_end_analysis", "T1_absq": "helix_end_analysis",
#                     "Tminus1_netq": "helix_end_analysis",
#                     "Tminus1_absq": "helix_end_analysis", "Tend_netq": "helix_end_analysis",
#                     "Tend_absq": "helix_end_analysis",
#                     "avg_all_frags": "fragment", "avg_best_frag": "fragment", "avg_best_frags": "fragment",
#                     "sum_best_frags": "fragment", "worstfrag": "fragment", "worst6frags": "fragment",
#                     "abd50_mean": "tertiary_motif", "abd50_min": "tertiary_motif", "dsc50_mean": "tertiary_motif",
#                     "dsc50_min": "tertiary_motif", "ssc50_mean": "tertiary_motif", "ssc50_min": "tertiary_motif",
#                     "hxl_tors": "not_sure", "fa_intra_rep_xover4": "not_sure", "fa_dun_rot": "not_sure",
#                     "fa_dun_dev": "not_sure", "fa_intra_elec": "not_sure",
#                     "fa_intra_atr_xover4": "not_sure", "contact_core_scn": "not_sure",
#                     "buns_bb_heavy": "not_sure", "fa_dun_semi": "not_sure", "n_polar_core": "not_sure",
#                     "contact_core_sasa": "not_sure", "bb": "not_sure", "buns_nonheavy": "not_sure",
#                     "buns_sc_heavy": "not_sure", "entropy": "not_sure"
#                     }
# feature_mappings = {k.lower(): v for k, v in feature_mappings.items()}
#
# df_graph = pd.DataFrame(columns=["Feature", "Left-Out-Group", "Pimportance", "Feature_Type"])
# left_out_groups = list(features_114k_RFR_stabilityscore.columns.values)
# left_out_groups.remove("Feature")
# left_out_groups.remove("mean")
# left_out_groups.remove("max")
# for feature in features_114k_RFR_stabilityscore['Feature']:
#     print(feature)
#     feature_type = feature_mappings[feature.lower()]
#     for group in left_out_groups:
#         pimportance = features_114k_RFR_stabilityscore.loc[
#             features_114k_RFR_stabilityscore["Feature"] == feature, group].item()
#         if pimportance > 0.001:
#             row_to_add = {"Feature": feature, "Left-Out-Group": group, "Pimportance": pimportance,
#                           "Feature_Type": feature_type}
#             df_graph = df_graph.append(row_to_add, ignore_index=True)
# print(df_graph)
#
# g = Graph()
# pg = g.create_projection_graph_from_df(df_graph, "Feature", "Left-Out-Group", "Pimportance",
#                                        label_col="Feature_Type", filename="feature_projection_with-type_0.001")
'''

def percent_overlap_two_lists_same_size(list1, list2):
    list_size = len(list1)
    if list_size != len(list2):
        raise ValueError("Both lists must be of the same size.")
    count = 0
    for x in list1:
        if x in list2:
            count = count + 1
    return float(count) / float(list_size)


def percent_overlap_heatmap(features_df, cols_to_compare, top_n=10):
    heatmap_df = pd.DataFrame(np.ones((len(cols_to_compare), len(cols_to_compare))), index=cols_to_compare,
                              columns=cols_to_compare)
    heatmap_combos = list(itertools.combinations(cols_to_compare, 2))

    for c1, c2 in heatmap_combos:
        c1_importances = list(features_df.sort_values(by=c1, ascending=False)[:top_n]['Feature'])
        c2_importances = list(features_df.sort_values(by=c2, ascending=False)[:top_n]['Feature'])
        percent_overlap = percent_overlap_two_lists_same_size(c1_importances, c2_importances)
        heatmap_df.loc[c1, c2] = percent_overlap
        heatmap_df.loc[c2, c1] = percent_overlap

    sns.clustermap(heatmap_df, annot=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
