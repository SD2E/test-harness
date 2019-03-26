import json
from datetime import datetime
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.model_selection import train_test_split

from harness.test_harness_class import TestHarness
from version import VERSION
from scripts_for_automation.perovskite_models_config import MODELS_TO_RUN


import warnings
warnings.filterwarnings("ignore")

PREDICTED_OUT = "predicted_out"
SCORE = "score"
# how many predictions from the test harness to send to submissions server
NUM_PREDICTIONS = 100

# todo: oops, committed this.  Need to revoke, but leaving for testing
AUTH_TOKEN = '4a8751b83c9744234367b52c58f4c46a53f5d0e0225da3f9c32ed238b7f82a69'


# compute md5 hash using small chunks
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_truncated_float(float_number, n=5):
    # By decision, we want to output reagent values as floats truncated (not rounded!) to 5 decimal places
    s = '{}'.format(float_number)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


def get_prediction_csvs(run_ids, predictions_csv_path=None):
    prediction_csv_paths = []
    if predictions_csv_path is None:
        runs_path = os.path.join('test_harness_results', 'runs')
        previous_runs = []
        for this_run_folder in os.listdir(runs_path):
            if this_run_folder.rsplit("_")[1] in run_ids:
                print('{} was kicked off by this TestHarness instance. Its results will be submitted.'.format(this_run_folder))
                prediction_csv_path = os.path.join(runs_path, this_run_folder, 'predicted_data.csv')
                if os.path.exists(prediction_csv_path):
                    print("file found: ", prediction_csv_path)
                    prediction_csv_paths.append(prediction_csv_path)
            else:
                previous_runs.append(this_run_folder)
        print('\nThe results for the following runs will not be submitted, '
              'because they are older runs that were not initiated by this TestHarness instance:'
              '\n{}\n'.format(previous_runs))
    else:
        prediction_csv_paths.append(predictions_csv_path)
    return prediction_csv_paths


def select_which_predictions_to_submit(predictions_df):
    # use binarized predictions, predict as 4
    subset = predictions_df.loc[predictions_df[PREDICTED_OUT] == 1, :]
    subset.loc[:, PREDICTED_OUT] = 4
    subset.sort_values(by=SCORE, inplace=True)
    return subset.head(NUM_PREDICTIONS)


def build_submissions_csvs_from_test_harness_output(prediction_csv_paths, crank_number, commit_id):
    submissions_paths = []
    for prediction_path in prediction_csv_paths:
        # todo: we need to know about what model this was for the notes field and such
        columns = {"dataset": "dataset",
                   "name": "name",
                   "_rxn_M_inorganic": "_rxn_M_inorganic",
                   "_rxn_M_organic": "_rxn_M_organic",
                   "_rxn_M_acid": "_rxn_M_acid",
                   "binarized_crystalscore_predictions": PREDICTED_OUT,
                   "binarized_crystalscore_prob_predictions": SCORE}
        df = pd.read_csv(prediction_path, comment='#')
        df = df.filter(columns.keys())
        df = df.rename(columns=columns)
        df['dataset'] = crank_number
        selected_predictions = select_which_predictions_to_submit(df)

        # fix formatting
        # truncate floats to 5 digits
        # for column in ['_rxn_M_inorganic', '_rxn_M_organic']:
        #     selected_predictions[column] = selected_predictions[column].apply(format_truncated_float)
        # 0-pad crank number if padding has been removed
        # selected_predictions['dataset'] = selected_predictions['dataset'].apply(lambda x: '{0:0>4}'.format(x))
        username = 'testharness'
        submission_template_filename = '_'.join([crank_number,
                                                 'train',
                                                 commit_id,
                                                 username]) + '.csv'
        submissions_file_path = os.path.join(os.path.dirname(prediction_path), submission_template_filename)

        print(submissions_file_path)
        selected_predictions.to_csv(submissions_file_path, index=False)
        submissions_paths.append(submissions_file_path)
    return submissions_paths


def submit_csv_to_escalation_server(submissions_file_path, crank_number, commit_id):
    print()

    test_harness_results_path = submissions_file_path.rsplit("/runs/")[0]
    this_run_results_path = submissions_file_path.rsplit("/", 1)[0]

    print(os.listdir(test_harness_results_path))

    leaderboard = pd.read_html(os.path.join(test_harness_results_path, 'custom_classification_leaderboard.html'))[0]
    leaderboard_entry_for_this_run = leaderboard.loc[leaderboard["Run ID"] == this_run_results_path.rsplit("/run_")[1]]

    model_name = leaderboard_entry_for_this_run["Model Name"].values[0]
    model_author = leaderboard_entry_for_this_run["Model Author"].values[0]
    model_description = leaderboard_entry_for_this_run["Model Description"].values[0]
    print(model_name)
    print(model_author)
    print(model_description)

    response = requests.post("http://escalation.sd2e.org/submission",
                             headers={'User-Agent': 'escalation'},
                             data={'crank': crank_number,
                                   'username': "test_harness_{}".format(VERSION),
                                   'expname': model_name,
                                   'githash': commit_id,
                                   # todo: add check to make sure that notes doesn't contain any commas
                                   'notes': "Model Author: {}; "
                                            "Model Description: {}; "
                                            "Submitted at {}".format(model_author, model_description,
                                                                     datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))},
                             files={'csvfile': open(submissions_file_path, 'rb')},
                             # timeout=60
                             )
    print("Submitted file to submissions server")
    return response, response.text


def get_crank_number_from_filename(training_data_filename):
    # Gets the challenge problem iteration number from the training data
    crank_number = os.path.basename(training_data_filename).split('.')[0]
    # should be of format ####
    assert len(crank_number) == 4
    return crank_number


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
    th = TestHarness(output_location=current_path, output_csvs_of_leaderboards=True)

    for model in MODELS_TO_RUN:
        th.run_custom(function_that_returns_TH_model=model, dict_of_function_parameters={},
                      training_data=train,
                      testing_data=test, data_and_split_description="test run on perovskite data",
                      cols_to_predict=col_to_predict,
                      feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                      feature_extraction=False, predict_untested_data=state_set,
                      index_cols=["dataset", "name", "_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid"]
                      )

    return th.list_of_this_instance_run_ids


def get_manifest_from_gitlab_api(commit_id, auth_token):
    headers = {"Authorization": "Bearer {}".format(auth_token)}
    # this is the API call for the versioned data repository.  It gets the raw data file.
    # 202 is the project id, derived from a previous call to the projects endpoint
    # we have hard code the file we are fetching (manifest/perovskite.manifest.yml), and vary the commit id to fetch
    gitlab_manifest_url = 'https://gitlab.sd2e.org/api/v4/projects/202/repository/files/manifest%2fperovskite.manifest.yml/raw?ref={}'.format(commit_id)
    response = requests.get(gitlab_manifest_url, headers=headers)
    if response.status_code == 404:
        raise KeyError("File perovskite manifest not found from Gitlab API for commit {}".format(commit_id))
    elif response.status_code == 403:
        raise RuntimeError("Unable to authenticate user with gitlab")
    perovskite_manifest = yaml.load(response.text)
    return perovskite_manifest


def get_git_hash_at_versioned_data_master_tip(auth_token):
    headers = {"Authorization": "Bearer {}".format(auth_token)}
    # this is the API call for the versioned data repository.  It gets the raw data file.
    # 202 is the project id, derived from a previous call to the projects endpoint
    # we have hard code the file we are fetching (manifest/perovskite.manifest.yml), and vary the commit id to fetch
    gitlab_manifest_url = 'https://gitlab.sd2e.org/api/v4//projects/202/repository/commits/master'
    response = requests.get(gitlab_manifest_url, headers=headers)
    if response.status_code == 404:
        raise KeyError("Unable to find metadata on master branch of versioned data repo via gitlab API")
    elif response.status_code == 403:
        raise RuntimeError("Unable to authenticate user with gitlab")
    gitlab_master_branch_metadata = json.loads(response.text)
    tip_commit_id = gitlab_master_branch_metadata["id"][:7]
    return tip_commit_id


def get_training_and_stateset_filenames(manifest):
    files_of_interest = {
        'perovskitedata': None,
        'stateset': None
    }
    print(files_of_interest.items())
    for file_name in manifest['files']:
        for file_type, existing_filename in files_of_interest.items():
            if file_name.endswith('{}.csv'.format(file_type)):
                if existing_filename:
                    raise KeyError(
                        "More than one file found in manifest of type {}.  Manifest files: {}".format(
                            file_type,
                            manifest['files'])
                    )
                else:
                    files_of_interest[file_type] = file_name
                    break
    for file_type, existing_filename in files_of_interest.items():
        assert existing_filename is not None, "No file found in manifest for type {}".format(file_type)
    return files_of_interest['perovskitedata'], files_of_interest['stateset']


if __name__ == '__main__':
    """
    NB: This script is for local testing, and is NOT what is run by the app.
    The test harness app runs in perovskite_test_harness.py
    """
    VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
    print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
    assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."

    training_data_filename = 'perovskite/perovskitedata/0021.perovskitedata.csv'
    # Reading in data from versioned-datasets repo.
    df = pd.read_csv(os.path.join(VERSIONED_DATA, training_data_filename),
                     comment='#',
                     low_memory=False)

    state_set = pd.read_csv(os.path.join(VERSIONED_DATA, 'perovskite/stateset/0021.stateset.csv'),
                            comment='#',
                            low_memory=False)

    list_of_run_ids = run_configured_test_harness_models_on_perovskites(df, state_set)

    # this uses current master commit on the origin
    commit_id = get_git_hash_at_versioned_data_master_tip(AUTH_TOKEN)

    crank_number = get_crank_number_from_filename(training_data_filename)
    prediction_csv_paths = get_prediction_csvs(run_ids=list_of_run_ids)
    submissions_paths = build_submissions_csvs_from_test_harness_output(prediction_csv_paths,
                                                                        crank_number,
                                                                        commit_id)
    for submission_path in submissions_paths:
        print("Submitting {} to escalation server".format(submission_path))
        response, response_text = submit_csv_to_escalation_server(submission_path, crank_number, commit_id)
        print("Submission result: {}".format(response_text))

# todo: round instead of truncate float
