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

from model_runner_instances.hamed_models.random_forest_regression import random_forest_regression, rfr_features

# SET PATH TO DATA FOLDER IN LOCALLY CLONED `versioned-datasets` REPO HERE:
# Note that if you clone the `versioned-datasets` repo at the same level as where you cloned the `protein-design` repo,
# then you can just use VERSIONED_DATASETS = os.path.join(Path(__file__).parents[2], 'versioned-datasets/data')
VERSIONED_DATA = os.path.join(Path(__file__).parents[2], 'versioned-datasets/data')
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

    combined_data = pd.read_csv(os.path.join(VERSIONED_DATA, 'aggregated_data/all_libs_cleaned.v0.aggregated_data.csv'),
                                comment='#', low_memory=False)

    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(combined_data[feature_cols])
    # normalized_df = combined_data.copy()
    # normalized_df[feature_cols] = scaler.transform(normalized_df[feature_cols])

    training_data, testing_data = train_test_split(combined_data, test_size=0.2, random_state=5,
                                                   stratify=combined_data[['topology', 'library']])

    training_data = training_data.sample(n=2000)
    testing_data = testing_data.sample(n=1000)

    # ------------------------------------------------------------------------------------------------------------------
    # Add the model runner instances that you want to run to the Test Harness here. Comment out any model runner
    # instances that you don't want to run.

    th.add_model_runner(rfr_features(training_data, testing_data))

    # Running Jed's Model, which requires GPU:
    # default_data_folder_path = os.path.join(PARENT, 'model_runner_data/default_model_runner_data/')
    # train_path = os.path.join(default_data_folder_path, 'consistent_normalized_training_data_v1.csv')
    # test_path = os.path.join(default_data_folder_path, 'consistent_normalized_testing_data_v1.csv')
    # untested_path = os.path.join(default_data_folder_path, 'normalized_and_cleaned_untested_designs_v1.csv')
    # th.add_model_runner(sequence_only_cnn(train_path, test_path, untested_path))

    # ------------------------------------------------------------------------------------------------------------------

    th.run_models()
    th.run_test_harness()

    ccl_path = os.path.join(output_dir, 'comparable_classification_leaderboard.html')
    crl_path = os.path.join(output_dir, 'comparable_regression_leaderboard.html')
    gcl_path = os.path.join(output_dir, 'general_classification_leaderboard.html')
    grl_path = os.path.join(output_dir, 'general_regression_leaderboard.html')

    # Build meta-leaderboard with DataTables styling
    TEMPLATES = os.path.join(PARENT, 'templates')
    index_path = os.path.join(output_dir, 'index.html')

    with open(index_path, 'w') as idx:
        with open(os.path.join(TEMPLATES, 'header.html.j2'), 'r') as hdr:
            for line in hdr:
                idx.write(line)

        for lb in (ccl_path, crl_path, gcl_path, grl_path):
            fname = os.path.basename(lb)
            classname = fname.replace('_leaderboard.html', '')
            heading = classname.replace('_', ' ').title()
            idx.write('\n<h2>{}</h2>\n'.format(heading))
            with open(lb, 'r') as tbl:
                for line in tbl:
                    idx.write(line)

        with open(os.path.join(TEMPLATES, 'footer.html.j2'), 'r') as ftr:
            for line in ftr:
                idx.write(line)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
