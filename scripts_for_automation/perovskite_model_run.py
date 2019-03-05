import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from scripts_for_automation.perovskite_models_config import MODELS_TO_RUN


def run_configured_test_harness_models_on_perovskites(train_set, state_set):
    print("Starting test harness initial perovskites run")
    all_cols = train_set.columns.tolist()
    # don't worry about _calc_ columns for now, but it's in the code so they get included once the data is available
    feature_cols = [c for c in all_cols if ("_rxn_" in c) or ("_feat_" in c) or ("_calc_" in c)]
    non_numerical_cols = (train_set.select_dtypes('object').columns.tolist())
    feature_cols = [c for c in feature_cols if c not in non_numerical_cols]

    # print(set(state_set.columns.tolist()).difference(set(feature_cols)))
    # print(set(feature_cols).difference(set(state_set.columns.tolist())))
    # remove _rxn_temperatureC_actual_bulk column from feature_cols because it doesn't exist in state_set
    feature_cols.remove("_rxn_temperatureC_actual_bulk")

    # create binarized crystal scores because Ian said to start with binary task
    # also multiclass support needs to be added to Test Harness
    conditions = [
        (train_set['_out_crystalscore'] == 1),
        (train_set['_out_crystalscore'] == 2),
        (train_set['_out_crystalscore'] == 3),
        (train_set['_out_crystalscore'] == 4),
    ]
    binarized_labels = [0, 0, 0, 1]
    train_set['binarized_crystalscore'] = np.select(conditions, binarized_labels)
    col_order = list(train_set.columns.values)
    col_order.insert(3, col_order.pop(col_order.index('binarized_crystalscore')))
    train_set = train_set[col_order]

    col_to_predict = 'binarized_crystalscore'

    train, test = train_test_split(train_set, test_size=0.2, random_state=5, stratify=train_set[['dataset']])

    # Test Harness use starts here:
    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}\n".format(current_path))
    th = TestHarness(output_location=current_path)

    for model in MODELS_TO_RUN:
        th.run_custom(function_that_returns_TH_model=model, dict_of_function_parameters={},
                      training_data=train,
                      testing_data=test, data_and_split_description="test run on perovskite data",
                      cols_to_predict=col_to_predict,
                      feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                      feature_extraction=False, predict_untested_data=state_set)


if __name__ == '__main__':
    VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
    print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
    assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."

    # Reading in data from versioned-datasets repo.
    df = pd.read_csv(os.path.join(VERSIONED_DATA, 'perovskite/perovskitedata/0018.perovskitedata.csv'),
                     comment='#',
                     low_memory=False)

    state_set = pd.read_csv(os.path.join(VERSIONED_DATA, 'perovskite/stateset/0018.stateset.csv'),
                            comment='#',
                            low_memory=False)

    run_configured_test_harness_models_on_perovskites(df, state_set)
