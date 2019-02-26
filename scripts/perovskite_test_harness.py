print("I am the perovskite test harness script")
"""
This script takes a commit_id, pulls the perovskite data for that commit, and runs the registered test harness models.

It's designed to run on TACC infrastructure, referencing file paths on their file system.

It is invoked by the test-harness-app Agave app.
"""


import argparse
import json
import yaml
import os

import requests


LOCAL_VERSIONED_DIR_PATH_CACHE = 'local_versioned_data_path_cache.json'
LOCAL_AUTH_TOKEN_CACHE = 'local_auth_token_cache.json'

#
# def make_cached_versioned_dir(versioned_data_path):
#     """
#     Creates a cache file to store the local path to the versioned data repo data directory
#     :param versioned_data_path:
#     :return:
#     """
#     print("Unable to find a versioned repo directory at:  {}".format(versioned_data_path))
#     # todo: separate out what runs on the agave app vs what can run locally. No input on app
#     versioned_data_dir = input("Enter the absolute path of your versioned-datasets folder containing data files (e.g., /Users/nick.leiby/versioned-datasets/data/): ")
#     with open(LOCAL_VERSIONED_DIR_PATH_CACHE, 'w') as fout:
#         json.dump({'path': versioned_data_dir}, fout)
#
#
# def get_versioned_data_dir():
#     """
#     Helper function to get the path to the versioned data dir depending on where the script is running
#     :return: local path to the versioned data repo data directory
#     """
#     versioned_data_path = '/work/projects/SD2E-Community/prod/data/versioned-dataframes/'
#     # check if on TACC
#     while not os.path.exists(versioned_data_path):
#         # check if running locally from cached local directory
#         if os.path.exists(LOCAL_VERSIONED_DIR_PATH_CACHE):
#             with open(LOCAL_VERSIONED_DIR_PATH_CACHE, 'r') as fin:
#                 versioned_data_path = json.load(fin).get('path')
#             if os.path.exists(versioned_data_path):
#                 return versioned_data_path
#             else:
#                 make_cached_versioned_dir(versioned_data_path)
#         else:
#             make_cached_versioned_dir(versioned_data_path)
#     return versioned_data_path
#

FILE_TYPE_OF_INTEREST = 'perovskitedata'
#
# def get_auth_token():
#     # # check if auth token is provided as environment variable
#     # while not os.environ.get('GITLAB_API_AUTH_TOKEN'):
#     #     # check if auth token is cached in file
#     #     if os.path.exists(LOCAL_AUTH_TOKEN_CACHE):
#     #         with open(LOCAL_VERSIONED_DIR_PATH_CACHE, 'r') as fin:
#     #             versioned_data_path = json.load(fin).get('path')
#     #         if os.path.exists(versioned_data_path):
#     #             return versioned_data_path
#     #         else:
#     #             make_cached_versioned_dir(versioned_data_path)
#     #     else:
#     #         make_cached_versioned_dir(versioned_data_path)
#     # return versioned_data_path
#     #
#
#     if os.environ.get('GITLAB_API_AUTH_TOKEN'):
#         auth_token = os.environ.get('GITLAB_API_AUTH_TOKEN')
#         print("Found auth token environment variable: %s" % auth_token)
#         return auth_token
#     #     # check if auth token is cached in file
#     # offer to cache auth token in file
#     print("Unable to find auth token")
#     return

# todo: oops, committed this.  Need to revoke, but leaving for testing
AUTH_TOKEN = '4a8751b83c9744234367b52c58f4c46a53f5d0e0225da3f9c32ed238b7f82a69'


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


def file_is_training_data(file_name):
    return file_name.endswith('{}.csv'.format(FILE_TYPE_OF_INTEREST))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', help='First 7 characters of versioned data commit hash')
    parser.add_argument('--gitlab_auth', help='gitlab auth token (see readme)')

    args = parser.parse_args()
    commit_id = args.commit or 'master'
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
    for file_name in perovskite_manifest['files']:
        if file_is_training_data(file_name):
            file_path = os.path.join(versioned_data_dir, perovskite_project_dir, file_name)
            print(file_path)
            # do we need to make sure we can't write/mess up any files here?
            # It'd sure be nice to have a read-only service account...
            with open(file_path, 'r') as fin:
                for i in range(10):
                    print(fin.readline())

    print("I finished the perovskite test harness script")
