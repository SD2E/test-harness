print("I am the perovskite test harness script")
"""
This script takes a commit_id, pulls the perovskite data for that commit, and runs the registered test harness models.

It's designed to run on TACC infrastructure, referencing file paths on their file system.

It is invoked by the test-harness-app Agave app.
"""
import argparse
import yaml
import os

import requests


PEROVSKITE_VERSIONED_DATA_DIR = '/work/projects/SD2E-Community/prod/data/versioned-dataframes/perovskite'

parser = argparse.ArgumentParser()
parser.add_argument('commit_hash', default='master', help='First 7 characters of versioned data commit hash')


def get_manifest_from_gitlab_api(commit_id):
    # todo: oops, committed this.  Need to revoke, but leaving for testing
    auth_token = '4a8751b83c9744234367b52c58f4c46a53f5d0e0225da3f9c32ed238b7f82a69'
    headers = {"Authorization": "Bearer {}".format(auth_token)}
    # this is the API call for the versioned data repository.  It gets the raw data file.
    # 202 is the project id, derived from a previous call to the projects endpoint
    # we have hard code the file we are fetching (manifest/fperovskite.manifest.yml), and vary the commit id to fetch
    gitlab_manifest_url = 'https://gitlab.sd2e.org/api/v4/projects/202/repository/files/manifest%2fperovskite.manifest.yml/raw?ref={}'.format(commit_id)
    response = requests.get(gitlab_manifest_url, headers=headers)
    if response.status_code == 404:
        raise KeyError("File perovskite manifest not found from Gitlab API for commit {}".format(commit_id))
    elif response.status_code == 403:
        raise RuntimeError("Unable to authenticate user with gitlab")
    perovskite_manifest = yaml.load(response.text)
    return perovskite_manifest


def file_is_training_data(file_name):
    return 'perovskitedata' in file_name


if __name__ == '__main__':
    args = parser.parse_args()
    commit_id = args.commit_hash
    perovskite_manifest = get_manifest_from_gitlab_api(commit_id)
    print(perovskite_manifest)

    # todo: shutils make copy of the file to local dir, use that
    for file_name in perovskite_manifest['files']:
        if file_is_training_data(file_name):
            file_path = os.path.join(PEROVSKITE_VERSIONED_DATA_DIR, file_name)
            print(file_path)
            # do we need to make sure we can't write/mess up any files here?
            # It'd sure be nice to have a read-only service account...
            with open(file_path, 'r') as fin:
                for i in range(10):
                    print(fin.readline())

    print("I finished the perovskite test harness script")
