"""
Runs 80-20 split for each model in MODELS_TO_RUN
 """

import os
from pathlib import Path


import yaml
from sklearn.model_selection import train_test_split, KFold

from harness.test_harness_class import TestHarness
from scripts_for_automation.perovskite_models_config import MODELS_TO_RUN
from scripts_for_automation.perovskite_model_run import (configure_input_df_for_test_harness,
                                                         get_latest_training_and_stateset_filenames,
                                                         get_crank_files,
                                                         get_crank_number_from_filename,
                                                         build_submissions_csvs_from_test_harness_output,
                                                         get_git_hash_at_versioned_data_master_tip,
                                                         get_prediction_csvs,
                                                         build_leaderboard_rows_dict)


import warnings


warnings.filterwarnings("ignore")

PREDICTED_OUT = "predicted_out"
SCORE = "score"
RANKING = "ranking"
# how many predictions from the test harness to send to submissions server
NUM_PREDICTIONS = 100

# todo: oops, committed this.  Need to revoke, but leaving for testing
AUTH_TOKEN = '4a8751b83c9744234367b52c58f4c46a53f5d0e0225da3f9c32ed238b7f82a69'


def run_configured_test_harness_models_on_80_20_splits(train_set, num_k=5, random_state=42, target_col='binarized_crystalscore'):

    # Test Harness use starts here:
    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}\n".format(current_path))
    th = TestHarness(output_location=current_path, output_csvs_of_leaderboards=True)

    train_set, feature_cols = configure_input_df_for_test_harness(train_set)
    kf = KFold(n_splits=num_k, random_state=random_state, shuffle=True)
    k_ind = 0
    for k_train_index, k_test_index in kf.split(train_set):
        train = train_set.iloc[k_train_index]
        test = train_set.iloc[k_test_index]

        for model in MODELS_TO_RUN:
            th.run_custom(function_that_returns_TH_model=model,
                          dict_of_function_parameters={},
                          training_data=train,
                          testing_data=test,
                          description="test run on perovskite data with 80/20 k fold k=%s" % k_ind,
                          target_cols=target_col,
                          feature_cols_to_use=feature_cols,
                          normalize=True,
                          feature_cols_to_normalize=feature_cols,
                          feature_extraction=False,
                          index_cols=["dataset", "name", "_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid"]
                          )
        k_ind += 1
    return th.list_of_this_instance_run_ids


def run_cranks(versioned_data_path, ):
    manifest_file = os.path.join(versioned_data_path, "manifest/perovskite.manifest.yml")
    with open(manifest_file) as f:
        manifest_dict = yaml.load(f)

    perovskite_data_folder_path = os.path.join(versioned_data_path, "data/perovskite")

    training_data_filename, state_set_filename = get_latest_training_and_stateset_filenames(manifest_dict)
    training_state_tuples = list(zip([training_data_filename], [state_set_filename]))

    print("\ntraining_state_tuples being passed to crank_runner:\n{}\n".format(training_state_tuples.copy()))
    for training_data_filename, state_set_filename in training_state_tuples:
        assert get_crank_number_from_filename(training_data_filename) == get_crank_number_from_filename(state_set_filename)
        training_data_path = os.path.join(perovskite_data_folder_path, training_data_filename)
        state_set_path = os.path.join(perovskite_data_folder_path, state_set_filename)
        training_data, state_set, crank_number = get_crank_files(training_data_path, state_set_path)

        list_of_run_ids = run_configured_test_harness_models_on_80_20_splits(training_data)
        # this uses current master commit on the origin
        prediction_csv_paths = get_prediction_csvs(run_ids=list_of_run_ids)


if __name__ == '__main__':

    VERSIONED_DATASETS = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets')

    print("Path to the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATASETS))
    print()
    assert os.path.isdir(VERSIONED_DATASETS), "The path you gave for VERSIONED_DATA does not exist."

    # set cranks equal to "latest", "all", or a string of format '0021' representing a specific crank number
    run_cranks(VERSIONED_DATASETS)

