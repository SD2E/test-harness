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
from th_model_instances.hamed_models.random_forest_classification import random_forest_classification

# SET PATH TO DATA FOLDER IN LOCALLY CLONED `versioned-datasets` REPO HERE:
# Note that if you clone the `versioned-datasets` repo at the same level as where you cloned the `protein-design` repo,
# then you can use VERSIONED_DATASETS = os.path.join(Path(__file__).resolve().parents[3], 'versioned-datasets/data')
VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[3], 'versioned-datasets/data')
print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
print()

PWD = os.getcwd()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)
RESULTSPATH = os.path.dirname(PARENT)
print("PWD:", PWD)
print("HERE:", HERE)
print("PARENT:", PARENT)
print("RESULTSPATH:", RESULTSPATH)
print()


def main():
    # Mohammed add start
    df = pd.read_csv('data/yeaststates/live_dead_dataframe.csv')
    print("Length of full DF", len(df))
    input_cols = ['FSC-A', 'SSC-A', 'BL1-A', 'RL1-A', 'FSC-H', 'SSC-H', 'BL1-H', 'RL1-H', 'FSC-W', 'SSC-W', 'BL1-W', 'RL1-W']
    output_cols = ["strain"]
    print("Size of filtered data", len(df))
    train, test = train_test_split(df, stratify=df['strain'], test_size=0.2, random_state=5)
    th = TestHarness(output_path=RESULTSPATH)

    rf_classification_model = random_forest_classification(n_estimators=500)
    th.add_custom_runs(test_harness_models=rf_classification_model, training_data=train, testing_data=test,
                       data_and_split_description="yeast_live_dead_dataframe",
                       cols_to_predict=output_cols,
                       feature_cols_to_use=input_cols, normalize=True, feature_cols_to_normalize=input_cols,
                       feature_extraction='rfpimp_permutation', predict_untested_data=False)
    # Mohammed add end
    th.execute_runs()


if __name__ == '__main__':
    main()
