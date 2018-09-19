import pandas as pd
import os
import glob
from builtins import any
from pathlib import Path

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', 200)


def combine_data(versioned_datasets_repo_path=os.path.join(Path(__file__).parents[3], 'versioned-datasets')):
    data_folder_path = os.path.join(versioned_datasets_repo_path, 'data')

    stab_scores_path = os.path.join(data_folder_path, 'experimental_stability_scores')
    metadata_path = os.path.join(data_folder_path, 'metadata')
    struct_metrics_path = os.path.join(data_folder_path, 'structural_metrics')

    stab_scores_files = glob.glob(os.path.join(stab_scores_path, '*.csv'))
    metadata_files = glob.glob(os.path.join(metadata_path, '*.csv'))
    struct_metrics_files = glob.glob(os.path.join(struct_metrics_path, '*.csv'))

    rocklin_rounds = ['rd1', 'rd2', 'rd3', 'rd4']
    stab_scores_files = [x for x in stab_scores_files if not any(y in x for y in rocklin_rounds)]
    metadata_files = [x for x in metadata_files if not any(y in x for y in rocklin_rounds)]
    struct_metrics_files = [x for x in struct_metrics_files if not any(y in x for y in rocklin_rounds)]

    libraries = ['Eva1', 'Eva2', 'Inna', 'Longxing', 'Rocklin', 'topology_mining_and_Longxing_chip_1',
                 'topology_mining_and_Longxing_chip_2', 'topology_mining_and_Longxing_chip_3']

    colnames = {}
    frames = []
    for idx, lib in enumerate(libraries):
        stab_scores_candidates = [x for x in stab_scores_files if lib + '.' in x]
        metadata_candidates = [x for x in metadata_files if lib + '.' in x]
        struct_metrics_candidates = [x for x in struct_metrics_files if lib + '.' in x]

        stab_scores_df = pd.read_csv(stab_scores_candidates[0], comment='#', low_memory=False)
        metadata_df = pd.read_csv(metadata_candidates[0], comment='#', low_memory=False)
        struct_metrics_df = pd.read_csv(struct_metrics_candidates[0], comment='#', low_memory=False)

        if lib is "Inna":
            struct_metrics_df.drop(['pdb_path', 'pdb_name'], axis=1, inplace=True)

        if lib is "topology_mining_and_Longxing_chip_2" or lib is "topology_mining_and_Longxing_chip_3":
            struct_metrics_df.drop(['topology'], axis=1, inplace=True)

        if lib is 'topology_mining_and_Longxing_chip_1' or lib is 'topology_mining_and_Longxing_chip_2' or \
                lib is 'topology_mining_and_Longxing_chip_3':
            struct_metrics_df.drop(['Unnamed: 0'], axis=1, inplace=True)

        if lib is 'Rocklin':
            stab_scores_df.drop(['ec50_pred_c', 'ec50_95ci_t', 'ec50_95ci_lbound_c', 'ec50_95ci_ubound_t',
                                 'ec50_pred_t', 'ec50_95ci_lbound_t', 'ec50_95ci_c', 'ec50_95ci_ubound_c'],
                                axis=1, inplace=True)
            stab_scores_df['stabilityscore_calibrated'] = stab_scores_df['stabilityscore']
            stab_scores_df['stabilityscore_calibrated_t'] = stab_scores_df['stabilityscore_t']
            stab_scores_df['stabilityscore_calibrated_c'] = stab_scores_df['stabilityscore_c']
            stab_scores_df['stabilityscore_cnn_calibrated'] = stab_scores_df['stabilityscore_cnn']
            stab_scores_df['stabilityscore_cnn_calibrated_t'] = stab_scores_df['stabilityscore_cnn_t']
            stab_scores_df['stabilityscore_cnn_calibrated_c'] = stab_scores_df['stabilityscore_cnn_c']

        merge_cols_1 = ['library', 'name']
        merge_cols_2 = ['library', 'name', 'sequence', 'dssp', 'description']
        df = metadata_df.merge(stab_scores_df, on=merge_cols_1, indicator='origin_1')
        df = df.merge(struct_metrics_df, on=merge_cols_2, indicator='origin_2')

        if df['origin_1'].eq('both').all() and df['origin_2'].eq('both').all():
            print("Rows matched up correctly during merging of 3 DataFrames for the {} library".format(lib))
        else:
            print("Rows DID NOT match up correctly during merging of 3 DataFrames for the {} library".format(lib))
        df.drop(['origin_1', 'origin_2'], inplace=True, axis=1)
        print("Merge of 3 DataFrames for the {} library resulted in a DataFrame of shape: {}".format(lib, df.shape))
        print()

        colnames[lib] = set(df.columns.values)
        frames.append(df)

    # check_if_cols_match_in_colnames_dict(colnames)

    final_df = pd.concat(frames, sort=False)
    return final_df


def check_if_cols_match_in_colnames_dict(colnames_dict):
    print("Comparing column names between different library DataFrames:\n")
    for libkey in colnames_dict:
        libcols = colnames_dict[libkey]
        others = colnames_dict.copy()
        others.pop(libkey)
        others_combined = set()
        for o in others.values():
            others_combined.update(o)
        print("Columns that exist in library {} but not in other libraries: {}".format(
            libkey, libcols.difference(others_combined)))
        print("Columns that exist in other libraries but not in library {}: {}".format(
            libkey, others_combined.difference(libcols)))
        print()


def combine_data_with_two_body_descriptors():
    pass
