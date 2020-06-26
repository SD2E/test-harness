import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.utils.names import Names
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
from scripts_for_automation.perovskite_model_run import get_crank_specific_training_and_stateset_filenames, get_crank_number_from_filename

VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."
print()


def main():
    # Testing multiclass classification using Perovskite data

    df = pd.read_csv(os.path.join(VERSIONED_DATA, "perovskite/perovskitedata/0014.perovskitedata.csv"), comment="#")
    print(df.head())

    # print(df["_out_crystalscore"].value_counts(dropna=False))
    df = df.loc[~df["_out_crystalscore"].isnull()]

    train, test = train_test_split(df, test_size=0.2, random_state=5)

    all_cols = df.columns.tolist()
    # don't worry about _calc_ columns for now, but it's in the code so they get included once the data is available
    feature_cols = [c for c in all_cols if ("_rxn_" in c) or ("_feat_" in c) or ("_calc_" in c)]
    non_numerical_cols = (df.select_dtypes('object').columns.tolist())
    feature_cols = [c for c in feature_cols if c not in non_numerical_cols]

    # TestHarness usage starts here, all code before this was just data input and pre-processing.
    current_path = os.getcwd()
    output_path = os.path.join(current_path, "multiclass_testing")
    print("initializing TestHarness object with output_location equal to {}".format(output_path))
    print()
    th = TestHarness(output_location=current_path)

    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
                  testing_data=test, description="testing multiclass classification on perovskite data",
                  cols_to_predict='_out_crystalscore', feature_cols_to_use=feature_cols, normalize=True,
                  feature_cols_to_normalize=feature_cols, feature_extraction=Names.ELI5_PERMUTATION)


if __name__ == '__main__':
    main()
