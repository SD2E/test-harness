"""
This script takes a commit_id, pulls the perovskite data for that commit, and runs the registered test harness models.

It's designed to run on TACC infrastructure, referencing file paths on their file system.

It is invoked by the test-harness-app Agave app.
"""


import argparse
import os

import pandas as pd

from scripts_for_automation.perovskite_model_run import (get_crank_number_from_filename,
                                                         run_configured_test_harness_models_on_perovskites,
                                                         submit_csv_to_escalation_server,
                                                         build_submissions_csvs_from_test_harness_output,
                                                         get_prediction_csvs, AUTH_TOKEN, get_manifest_from_gitlab_api,
                                                         get_git_hash_at_versioned_data_master_tip,
                                                         get_training_and_stateset_filenames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', help='First 7 characters of versioned data commit hash')
    parser.add_argument('--gitlab_auth', help='gitlab auth token (see readme)')

    args = parser.parse_args()
    commit_id = args.commit or get_git_hash_at_versioned_data_master_tip(AUTH_TOKEN)

    # auth_token = args.gitlab_auth or get_auth_token()
    auth_token = args.gitlab_auth or AUTH_TOKEN

    print('auth token: %s' % auth_token)
    print(commit_id)
    perovskite_manifest = get_manifest_from_gitlab_api(commit_id, auth_token)
    print(perovskite_manifest)
    perovskite_project_dir = perovskite_manifest['project name']
    # versioned_data_dir = get_versioned_data_dir()
    versioned_data_dir = '/work/projects/SD2E-Community/prod/data/versioned-dataframes/'

    # todo: shutils make copy of the file to local dir, use that
    # do we need to make sure we can't write/mess up any files here?
    # It'd sure be nice to have a read-only service account...
    training_data_filename, stateset_filename = get_training_and_stateset_filenames(perovskite_manifest)
    print(training_data_filename, stateset_filename)
    training_data_df = pd.read_csv(os.path.join(versioned_data_dir, perovskite_project_dir, training_data_filename),
                                   comment='#',
                                   low_memory=False)
    print(training_data_df.head())
    stateset_df = pd.read_csv(os.path.join(versioned_data_dir, perovskite_project_dir, stateset_filename),
                              comment='#',
                              low_memory=False)
    print(stateset_df.head())

    # todo: this should return ranked predictions
    list_of_run_ids = run_configured_test_harness_models_on_perovskites(train_set=training_data_df, state_set=stateset_df)

    crank_number = get_crank_number_from_filename(training_data_filename)
    prediction_csv_paths = get_prediction_csvs(list_of_run_ids)
    submissions_paths = build_submissions_csvs_from_test_harness_output(prediction_csv_paths,
                                                                        crank_number,
                                                                        commit_id)
    for submission_path in submissions_paths:
        print("Submitting {} to escalation server".format(submission_path))
        response, response_text = submit_csv_to_escalation_server(submission_path, crank_number, commit_id)
        print("Submission result: {}".format(response_text))


