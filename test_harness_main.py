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

from model_runner_instances.hamed_models.random_forest_classification import random_forest_classification





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

    Returns: TestHarnessModel
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
    if 'output' in args and args.output is not None:
        output_dir = os.path.join(PWD, args.output)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception:
                raise
    else:
        output_dir = PARENT
    #Mohammed add start
    df = pd.read_excel('data/yeaststates/live_dead_dataframe.csv')
    print("Length of full DF", len(df))
    input_cols = ['FSC-A','SSC-A','BL1-A','RL1-A','FSC-H','SSC-H','BL1-H','RL1-H','FSC-W','SSC-W','BL1-W','RL1-W']
    output_cols = ["strain"]
    print("Size of filtered data",len(df))
    train, test = train_test_split(df, stratify=df['strain'], test_size=0.2, random_state=5)
    th = TestHarness(output_path=output_dir)

    rf_classification_model = random_forest_classification(n_estimators=500)
    th.add_custom_runs(test_harness_models=rf_classification_model, training_data=train, testing_data=test,
                       data_and_split_description="just testing things out!",
                       cols_to_predict=output_cols,
                       feature_cols_to_use=input_cols, normalize=True, feature_cols_to_normalize=input_cols,
                       feature_extraction=False, predict_untested_data=False)
    #Mohammed add end
    th.execute_runs()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
