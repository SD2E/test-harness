import argparse
import datetime as dt
import os
import importlib
import types
import pandas as pd
from tabulate import tabulate
from test_harness_class import TestHarness


#from model_runner_instances.hamed_models.rocklin_models import logistic_classifier_topology_general_all_features
from model_runner_instances.hamed_models.random_forest_regression import random_forest_regression, rfr_features

PWD = os.getcwd()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)

print("PWD:", PWD)
print("HERE:", HERE)
print("PARENT:", PARENT)


parser = argparse.ArgumentParser()
# Default behavior is to write out relative
# to test_harness. Passing output will cause
# writes to occur to a path relative to the current working directory
parser.add_argument('--output', required=False,
                    help='Output directory')


def model_runner_by_name(model_runner_path,
                         module_base_path='model_runner_instances'):
    """
    Instantiate an instance of model_runner by path

    Returns: ModelRunner
    Raises: Exception
    """
    try:
        path_parts = model_runner_path.split('.')
        func_name = path_parts[-1]
        func_module_parent_path = module_base_path + '.' + \
            '.'.join(path_parts[:-1])
        func_module = importlib.import_module(func_module_parent_path)
        named_meth = getattr(func_module, func_name)
        if callable(named_meth) and isinstance(named_meth,
                                               types.FunctionType):
            model_runner_instance = named_meth()
            return model_runner_instance
    # TODO: More granular Exception handling
    except Exception:
        raise


def main(args):

    model_list = []

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

    training_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v2-108k/normalized_data_v2_train.csv')
    testing_data = pd.read_csv('/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v2-108k/normalized_data_v2_test.csv')


    # training_data = os.path.join(PWD, 'versioned_data/asap/consistent_training_data_v1.asap.csv')
    # testing_data = os.path.join(PWD, 'versioned_data/asap/consistent_testing_data_v1.asap.csv')

    # training_data = pd.read_csv('/work/projects/SD2E-Community/prod/data/protein-design/versioned_data/asap/consistent_training_data_v1.asap.csv')
    # testing_data = pd.read_csv('/work/projects/SD2E-Community/prod/data/protein-design/versioned_data/asap/consistent_testing_data_v1.asap.csv')

    training_data = training_data.sample(n=2000)
    testing_data = testing_data.sample(n=1000)


    # ------------------------------------------------------------------------------------------------------------------
    # Add the model runner instances that you want to run to the Test Harness here. Comment out any model runner
    # instances that you don't want to run.

    th.add_model_runner(rfr_features(training_data, testing_data))

    # Running Jed's Model, which requires GPU:
    # default_data_folder_path = os.path.join(PARENT, 'model_runner_data/default_model_runner_data/')
    # train_path = os.path.join(default_data_folder_path, 'consistent_normalized_training_data_v1.csv')
    # test_path = os.path.join(default_data_folder_path, 'consistent_normalized_testing_data_v1.csv')
    # untested_path = os.path.join(default_data_folder_path, 'normalized_and_cleaned_untested_designs_v1.csv')
    # th.add_model_runner(sequence_only_cnn(train_path, test_path, untested_path))

    # ------------------------------------------------------------------------------------------------------------------

    th.run_models()
    th.run_test_harness()

    ccl_path = os.path.join(output_dir, 'comparable_classification_leaderboard.html')
    crl_path = os.path.join(output_dir, 'comparable_regression_leaderboard.html')
    gcl_path = os.path.join(output_dir, 'general_classification_leaderboard.html')
    grl_path = os.path.join(output_dir, 'general_regression_leaderboard.html')
    # print(tabulate(pd.read_html(ccl_path)[0], headers='keys', tablefmt='grid'))
    # print(tabulate(pd.read_html(crl_path)[0], headers='keys', tablefmt='grid'))
    # print(tabulate(pd.read_html(gcl_path)[0], headers='keys', tablefmt='grid'))
    # print(tabulate(pd.read_html(grl_path)[0], headers='keys', tablefmt='grid'))

    # Build meta-leaderboard with DataTables styling
    TEMPLATES = os.path.join(PARENT, 'templates')
    index_path = os.path.join(output_dir, 'index.html')

    with open(index_path, 'w') as idx:
        with open(os.path.join(TEMPLATES, 'header.html.j2'), 'r') as hdr:
            for line in hdr:
                idx.write(line)

        for lb in (ccl_path, crl_path, gcl_path, grl_path):
            fname = os.path.basename(lb)
            classname = fname.replace('_leaderboard.html', '')
            heading = classname.replace('_', ' ').title()
            idx.write('\n<h2>{}</h2>\n'.format(heading))
            with open(lb, 'r') as tbl:
                for line in tbl:
                    idx.write(line)

        with open(os.path.join(TEMPLATES, 'footer.html.j2'), 'r') as ftr:
            for line in ftr:
                idx.write(line)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
