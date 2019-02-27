import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification


def initial_perovskites_run(df):
    print("Starting test harness initial perovskites run")
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if ("_rxn_" in c) or ("_feat_" in c)]
    non_numerical_cols = (df.select_dtypes('object').columns.tolist())
    feature_cols = [c for c in feature_cols if c not in non_numerical_cols]

    # create binarized crystal scores because Ian said to start with binary task
    # also multiclass support needs to be added to Test Harness
    conditions = [
        (df['_out_crystalscore'] == 1),
        (df['_out_crystalscore'] == 2),
        (df['_out_crystalscore'] == 3),
        (df['_out_crystalscore'] == 4),
    ]
    binarized_labels = [0, 0, 0, 1]
    df['binarized_crystalscore'] = np.select(conditions, binarized_labels)
    col_order = list(df.columns.values)
    col_order.insert(3, col_order.pop(col_order.index('binarized_crystalscore')))
    df = df[col_order]
    print(df)

    col_to_predict = 'binarized_crystalscore'

    train, test = train_test_split(df, test_size=0.2, random_state=5, stratify=df[['dataset']])

    # Test Harness use starts here:
    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}\n".format(current_path))
    th = TestHarness(output_location=current_path)

    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={},
                  training_data=train,
                  testing_data=test, data_and_split_description="test run on perovskite data",
                  cols_to_predict=col_to_predict,
                  feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                  feature_extraction=False, predict_untested_data=False)


if __name__ == '__main__':
    VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
    print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
    assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."


    # Reading in data from versioned-datasets repo.
    df = pd.read_csv(
        os.path.join(VERSIONED_DATA, 'perovskite/experimental_run/2017-07-20-Final.experimental_run.csv'),
        comment='#',
        low_memory=False)

    initial_perovskites_run(df)

