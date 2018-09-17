import argparse
import os
import sys
import pandas as pd
from tabulate import tabulate
from test_harness.test_harness import TestHarness


#from model_runner_instances.hamed_models.rocklin_models import logistic_classifier_topology_general_all_features
from model_runner_instances.hamed_models.random_forest_regression import random_forest_regression, rfr_features

PWD = os.getcwd()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
# Default behavior is to write out relative
# to test_harness. Passing output will cause
# writes to occur to a path relative to the current working directory
parser.add_argument('--output', required=False,
                    help='Output directory')


pd.options.display.float_format = '{:.3f}'.format

def main(args):

    if 'output' in args and args.output is not None:
        output_dir = os.path.join(PWD, args.output)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception:
                    raise
    else:
        output_dir = PARENT

    th = TestHarness(output_path=output_dir)



    # training_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v0-15k/normalized_data_v0_train.csv')
    # testing_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v0-15k/normalized_data_v0_test.csv')

    # training_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v1-82k/normalized_data_v1_train.csv')
    # testing_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v1-82k/normalized_data_v1_test.csv')

    # training_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v2-108k/normalized_data_v2_train.csv')
    # testing_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v2-108k/normalized_data_v2_test.csv')


    # training_data = os.path.join(PWD, 'versioned_data/asap/consistent_training_data_v1.asap.csv')
    # testing_data = os.path.join(PWD, 'versioned_data/asap/consistent_testing_data_v1.asap.csv')
    # training_data = training_data.sample(n=2000)
    # testing_data = testing_data.sample(n=1000)





    #sys.exit()

    training_data = pd.read_csv('/work/projects/SD2E-Community/prod/data/protein-design/versioned_data/asap/consistent_training_data_v1.asap.csv')
    testing_data = pd.read_csv('/work/projects/SD2E-Community/prod/data/protein-design/versioned_data/asap/consistent_testing_data_v1.asap.csv')


    # ------------------------------------------------------------------------------------------------------------------
    # Add the model runner instances that you want to run to the Test Harness here. Comment out any model runner
    # instances that you don't want to run.

    th.add_model_runner(rfr_features(training_data, testing_data))


    # ------------------------------------------------------------------------------------------------------------------

    # th.run_models()
    # th.run_test_harness()
    th.run_models_on_different_splits(performance_output_path='splits/feature_importances/asdf1.csv',
                                      features_output_path='splits/feature_importances/asdf2.csv')


    '''
    hughs_groupings = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/test_harness/'
                                  'splits/leave_one_out_split_groupings.csv')

    print(hughs_groupings)

    print(set(training_data['library']))
    
    th.run_models_on_custom_splits(hughs_groupings, performance_output_path='splits/feature_importances/permut_rfr_k_performance.csv',
                                   features_output_path='splits/feature_importances/permut_rfr_k_features.csv')

    '''



    # ccl_path = os.path.join(output_dir, 'comparable_classification_leaderboard.html')
    # crl_path = os.path.join(output_dir, 'comparable_regression_leaderboard.html')
    # gcl_path = os.path.join(output_dir, 'general_classification_leaderboard.html')
    # grl_path = os.path.join(output_dir, 'general_regression_leaderboard.html')
    # print(tabulate(pd.read_html(ccl_path)[0], headers='keys', tablefmt='grid'))
    # print(tabulate(pd.read_html(crl_path)[0], headers='keys', tablefmt='grid'))
    # print(tabulate(pd.read_html(gcl_path)[0], headers='keys', tablefmt='grid'))
    # print(tabulate(pd.read_html(grl_path)[0], headers='keys', tablefmt='grid'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
