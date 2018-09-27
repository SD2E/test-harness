import argparse
import datetime as dt
import os
import importlib
import types
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from versioned_data_scripts.combine_versioned_data_v0 import combine_data
from sklearn.model_selection import train_test_split
from test_harness_class import TestHarness

from model_runner_instances.hamed_models.random_forest_regression import rfr_features
from model_runner_instances.jed_models.sequence_cnn import sequence_only_cnn
from model_runner_instances.hamed_models.rocklin_models import linear_regression_topology_general_all_features as linreg

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

    new_ss_file_names = ['Eva1.Rocklin_calibrated_experimental_stability_scores.v4.csv',
                         'Eva2.Rocklin_calibrated_experimental_stability_scores.v4.csv',
                         'Inna.Rocklin_calibrated_experimental_stability_scores.v4.csv',
                         'Longxing.Rocklin_calibrated_experimental_stability_scores.v4.csv',
                         'Rocklin.Rocklin_calibrated_experimental_stability_scores.csv',
                         'topology_mining_and_Longxing_chip_1.Rocklin_calibrated_experimental_stability_scores.v4.csv',
                         'topology_mining_and_Longxing_chip_2.Rocklin_calibrated_experimental_stability_scores.v4.csv',
                         'topology_mining_and_Longxing_chip_3.Rocklin_calibrated_experimental_stability_scores.v3.csv']
    frames = []
    for nssf in new_ss_file_names:
        df = pd.read_csv(os.path.join(VERSIONED_DATA, 'experimental_stability_scores/', nssf), comment='#',
                         low_memory=False)
        df_new_ss = df[['library', 'name', 'stabilityscore_calibrated', 'stabilityscore_calibrated_t',
                        'stabilityscore_calibrated_c']]
        frames.append(df_new_ss)
    francis_new_calibrated_ss = pd.concat(frames)
    francis_new_calibrated_ss.rename(columns={'stabilityscore_calibrated': 'stabilityscore_calibrated_v2',
                                              'stabilityscore_calibrated_t': 'stabilityscore_calibrated_t_v2',
                                              'stabilityscore_calibrated_c': 'stabilityscore_calibrated_c_v2'},
                                     inplace=True)

    combined_data = pd.read_csv(os.path.join(VERSIONED_DATA, 'aggregated_data/all_libs_cleaned.v0.aggregated_data.csv'),
                                comment='#', low_memory=False)

    combined_data = combined_data.merge(francis_new_calibrated_ss, on=['library', 'name'])

    combined_data['library_original'] = combined_data['library']
    combined_data['library'] = combined_data['library'].replace({"topology_mining_and_Longxing_chip_1": "t_l_untested",
                                                                 "topology_mining_and_Longxing_chip_2": "t_l_untested",
                                                                 "topology_mining_and_Longxing_chip_3": "t_l_untested"})

    # Changing the order of columns in combined_data
    col_order = list(combined_data.columns.values)
    col_order.insert(2, col_order.pop(col_order.index('library_original')))
    combined_data = combined_data[col_order]

    data_RD_16k = combined_data.loc[combined_data['library_original'] == 'Rocklin'].copy()
    data_RD_BL_81k = combined_data.loc[
        combined_data['library_original'].isin(['Rocklin', 'Eva1', 'Eva2', 'Inna', 'Longxing'])].copy()
    data_RD_BL_TA1R1_105k = combined_data.loc[
        combined_data['library_original'].isin(
            ['Rocklin', 'Eva1', 'Eva2', 'Inna', 'Longxing', 'topology_mining_and_Longxing_chip_1',
             'topology_mining_and_Longxing_chip_2'])].copy()
    data_RD_BL_TA1R1_KJ_114k = combined_data.loc[
        combined_data['library_original'].isin(
            ['Rocklin', 'Eva1', 'Eva2', 'Inna', 'Longxing', 'topology_mining_and_Longxing_chip_1',
             'topology_mining_and_Longxing_chip_2', 'topology_mining_and_Longxing_chip_3'])].copy()

    # Grouping Data
    grouping_df = pd.read_csv(os.path.join(VERSIONED_DATA, 'protein_groupings/v4_data.hugh.v1.protein_groupings.csv'),
                              comment='#', low_memory=False)
    grouping_df['library'] = grouping_df['library'].replace({"longxing_untested": "t_l_untested",
                                                             "topmining_untested": "t_l_untested"})
    print(grouping_df)

    # Change these values for different models/col_to_predict/data
    # model options: "RFR", "CNN", "lingreg"
    # col_to_predict options: "stabilityscore", "stabilityscore_calibrated", "stabilityscore_cnn",
    #                         "stabilityscore_cnn_calibrated", "stabilityscore_calibrated_v2"
    # data_set_description options: "16k", "81k", "105k", "114k"
    # --------------
    model = "CNN"
    col_to_predict = "stabilityscore"
    data_set_description = "114k"
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
        th.add_model_runner(rfr_features(use_this_data, pd.DataFrame(), col_to_predict=col_to_predict,
                                         data_set_description=data_set_description,
                                         train_test_split_description="leave-one-group-out"))
        should_i_normalize = True
        get_pimportances = True
    elif model == "CNN":
        th.add_model_runner(sequence_only_cnn(use_this_data, pd.DataFrame(), col_to_predict=col_to_predict,
                                              data_set_description=data_set_description,
                                              train_test_split_description="leave-one-group-out"))
        should_i_normalize = False
        get_pimportances = False
    elif model == "linreg":
        th.add_model_runner(linreg(use_this_data, pd.DataFrame(), col_to_predict=col_to_predict,
                                   data_set_description=data_set_description,
                                   train_test_split_description="leave-one-group-out"))
        should_i_normalize = True
        get_pimportances = False
    else:
        raise ValueError("for this temporary analysis script, model must equal RFR, CNN, or Linreg")

    th.run_models_on_custom_splits(grouping_df=grouping_df, performance_output_path=perf_path,
                                   features_output_path=feat_path, normalize=should_i_normalize,
                                   get_pimportances=get_pimportances)

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
