import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from test_harness.test_harness_class import TestHarness
from test_harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from test_harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
from pysd2cat.data import pipeline

# SET PATH TO DATA FOLDER IN LOCALLY CLONED `versioned-datasets` REPO HERE:
# Note that if you clone the `versioned-datasets` repo at the same level as where you cloned the `protein-design` repo,
# then you can use VERSIONED_DATASETS = os.path.join(Path(__file__).resolve().parents[3], 'versioned-datasets/data')
#VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[3], 'versioned-datasets/data')
#print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
#print()

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
    data_dir = '/work/projects/SD2E-Community/prod/data/uploads/'
    print("Building Live/Dead Control Dataframe...")
    df = pipeline.get_dataframe_for_live_dead_classifier(data_dir)

    print("Length of full DF", len(df))
    input_cols = ['FSC-A', 'SSC-A', 'BL1-A', 'RL1-A', 'FSC-H', 'SSC-H', 'BL1-H', 'RL1-H', 'FSC-W', 'SSC-W', 'BL1-W', 'RL1-W']
    output_cols = ["class_label"]
    print("Size of filtered data", len(df))
    train, test = train_test_split(df, stratify=df["class_label"], test_size=0.2, random_state=5)
    examples_folder_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(examples_folder_path))
    print()
    th = TestHarness(output_location=examples_folder_path)

    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
                  testing_data=test, data_and_split_description="example custom run on Rocklin data",
                  cols_to_predict="class_label",
                  feature_cols_to_use=input_cols, normalize=True, feature_cols_to_normalize=input_cols,
                  feature_extraction=False, predict_untested_data=False)


if __name__ == '__main__':
    main()
