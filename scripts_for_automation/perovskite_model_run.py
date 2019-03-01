from datetime import datetime
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from harness.test_harness_class import TestHarness
from scripts_for_automation.perovskite_models_config import MODELS_TO_RUN


PREDICTED_OUT = "predicted_out"
SCORE = "score"
# how many predictions from the test harness to send to submissions server
NUM_PREDICTIONS = 100

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
    return '.'.join([i, (d+'0'*n)[:n]])


def get_prediction_csvs(predictions_csv_path=None):
    prediction_csv_paths = []
    if predictions_csv_path is None:
        runs_path = os.path.join('test_harness_results', 'runs')
        for this_run_path in os.listdir(runs_path):
            prediction_csv_path = os.path.join(runs_path, this_run_path, 'predicted_data.csv')
            if os.path.exists(prediction_csv_path):
                print("file found: ", prediction_csv_path)
                prediction_csv_paths.append(prediction_csv_path)
    else:
        prediction_csv_paths.append(predictions_csv_path)
    return prediction_csv_paths


def select_which_predictions_to_submit(predictions_df):
    # use binarized predictions, predict as 4
    subset = predictions_df.loc[predictions_df[PREDICTED_OUT] == 1, :]
    subset.loc[:, PREDICTED_OUT] = 4
    subset.sort_values(by=SCORE, inplace=True)
    return subset.head(NUM_PREDICTIONS)


def build_submissions_csvs_from_test_harness_output(prediction_csv_paths, stateset_hash, crank_number, commit_id):
    submissions_paths = []
    for prediction_path in prediction_csv_paths:
        # todo: we need to know about what model this was for the notes field and such
        columns = {"dataset": "dataset",
                   "name": "name",
                   "_rxn_M_inorganic": "_rxn_M_inorganic",
                   "_rxn_M_organic": "_rxn_M_organic",
                   "binarized_crystalscore_predictions": PREDICTED_OUT,
                   "binarized_crystalscore_prob_predictions": SCORE}
        df = pd.read_csv(prediction_path, comment='#')
        df = df.filter(columns.keys())
        df = df.rename(columns=columns)
        df['dataset'] = stateset_hash[:11]
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
        selected_predictions.to_csv(submissions_file_path, index=False)
        submissions_paths.append(submissions_file_path)
    return submissions_paths


def submit_csv_to_escalation_server(submissions_file_path, crank_number):
    response = requests.post("http://escalation.sd2e.org/submission",
                             headers={'User-Agent': 'escalation'},
                             data={'crank': crank_number,
                                   'username': "test_harness",
                                   'expname': "Test harness automated run",
                                   'notes': "Submitted at at {}".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))},
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
    th = TestHarness(output_location=current_path)

    for model in MODELS_TO_RUN:
        th.run_custom(function_that_returns_TH_model=model, dict_of_function_parameters={},
                      training_data=train,
                      testing_data=test, data_and_split_description="test run on perovskite data",
                      cols_to_predict=col_to_predict,
                      feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                      feature_extraction=False, predict_untested_data=state_set,
                      index_cols=[
                          "dataset",
                          "name",
                          "_rxn_M_inorganic",
                          "_rxn_M_organic"
                      ]
                      )

if __name__ == '__main__':
    VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')
    print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
    assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."

    training_data_filename = 'perovskite/perovskitedata/0018.perovskitedata.csv'
    # # Reading in data from versioned-datasets repo.
    df = pd.read_csv(os.path.join(VERSIONED_DATA, training_data_filename),
                     comment='#',
                     low_memory=False)

    state_set = pd.read_csv(os.path.join(VERSIONED_DATA, 'perovskite/stateset/0018.stateset.csv'),
                            comment='#',
                            low_memory=False)

    run_configured_test_harness_models_on_perovskites(df, state_set)

    stateset_hash = md5(os.path.join(VERSIONED_DATA, 'perovskite/stateset/0018.stateset.csv'))
    commit_id = 'abc12345678'
    crank_number = get_crank_number_from_filename(training_data_filename)
    prediction_csv_paths = get_prediction_csvs()
    submissions_paths = build_submissions_csvs_from_test_harness_output(prediction_csv_paths,
                                                                        stateset_hash,
                                                                        crank_number,
                                                                        commit_id)
    for submission_path in submissions_paths:
        print("Submitting {} to escalation server".format(submission_path))
        submit_csv_to_escalation_server(submission_path, crank_number)


# todo: remove hash requirement from submission
# round instead of truncate float
