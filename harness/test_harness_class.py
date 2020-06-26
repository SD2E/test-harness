from collections import defaultdict
from datetime import datetime
import os
import json
import time

import pandas as pd
import matplotlib.pyplot as plt
from six import string_types
from statistics import mean
import joblib
from copy import copy, deepcopy

from harness.run_classes import _BaseRun
from harness.test_harness_models_abstract_classes import ClassificationModel, RegressionModel
from harness.unique_id import get_id
from harness.utils.names import Names
from harness.utils.object_type_modifiers_and_checkers import is_list_of_strings, make_list_if_not_list

plt.switch_backend('agg')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
# CSS classes applied to the Pandas Dataframes when written as HTML
css_classes = ["table-bordered", "table-striped", "table-compact"]

PWD = os.getcwd()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)
DEFAULT_DATA_PATH = os.path.join(PWD, 'versioned_data/asap/')
OUTPUT = Names.NORMAL_OUTPUT

'''
NOTE: If a class variable is going to be modified (e.g. feature_cols_to_use is modified by sparse col functionality),
then you must make sure that a COPY of the variable is passed in! Otherwise the original variable will be modified too, leading to issues.
'''


# TODO: add ran-by (user) column to leaderboards
# TODO: add md5hashes of data to leaderboard as sorting tool
# TODO: add cross validation
# TODO: if test set doesn't include target_col, carry out prediction instead?
# TODO: add more checks for correct inputs using assert
# TODO: add filelock or writing-scheduler so leaderboards are not overwritten at the same time. Might need to use SQL
# TODO: separate data description from split description


class TestHarness:
    def __init__(self, output_location=os.path.dirname(os.path.realpath(__file__)), output_csvs_of_leaderboards=False,
                 compress_large_csvs=False):
        # Note: loo stands for leave-one-out
        self.output_path = output_location
        self.output_csvs_of_leaderboards = output_csvs_of_leaderboards
        self.results_folder_path = os.path.join(self.output_path, Names.TEST_HARNESS_RESULTS_DIR)
        self.runs_folder_path = os.path.join(self.results_folder_path, Names.RUNS_DIR)
        if not os.path.exists(self.results_folder_path):
            os.makedirs(self.results_folder_path, exist_ok=True)
        if not os.path.exists(self.runs_folder_path):
            os.makedirs(self.runs_folder_path, exist_ok=True)
        assert isinstance(compress_large_csvs, bool), "compress_large_csvs must be True or False"
        self.compress_large_csvs = compress_large_csvs

        # add metrics here:
        self.classification_metrics = [Names.NUM_CLASSES, Names.ACCURACY, Names.BALANCED_ACCURACY, Names.AUC_SCORE,
                                       Names.AVERAGE_PRECISION, Names.F1_SCORE, Names.PRECISION, Names.RECALL]
        self.mean_classification_metrics = ["Mean " + cm for cm in self.classification_metrics]
        self.regression_metrics = [Names.R_SQUARED, Names.RMSE]
        self.mean_regression_metrics = ["Mean " + rm for rm in self.regression_metrics]

        self.metric_to_sort_classification_results_by = Names.AVERAGE_PRECISION
        self.metric_to_sort_regression_results_by = Names.R_SQUARED

        custom_cols_1 = [Names.RUN_ID, Names.DATE, Names.TIME, Names.MODEL_NAME, Names.MODEL_AUTHOR]
        custom_cols_2 = [Names.SAMPLES_IN_TRAIN, Names.SAMPLES_IN_TEST, Names.MODEL_DESCRIPTION, Names.COLUMN_PREDICTED,
                         Names.NUM_FEATURES_USED, Names.DESCRIPTION, Names.NORMALIZED, Names.NUM_FEATURES_NORMALIZED,
                         Names.FEATURE_EXTRACTION, Names.WAS_UNTESTED_PREDICTED]
        self.custom_classification_leaderboard_cols = custom_cols_1 + self.classification_metrics + custom_cols_2
        self.custom_regression_leaderboard_cols = custom_cols_1 + self.regression_metrics + custom_cols_2

        loo_cols_1 = [Names.LOO_ID] + custom_cols_1
        loo_cols_2 = custom_cols_2[:]
        loo_cols_2.remove(Names.WAS_UNTESTED_PREDICTED)
        loo_cols_2.insert(5, Names.TEST_GROUP)
        self.loo_full_classification_leaderboard_cols = loo_cols_1 + self.classification_metrics + loo_cols_2
        self.loo_full_regression_leaderboard_cols = loo_cols_1 + self.regression_metrics + loo_cols_2

        summarized_cols_1 = loo_cols_1[:]
        summarized_cols_1.remove(Names.RUN_ID)
        summarized_cols_2 = [Names.MODEL_DESCRIPTION, Names.COLUMN_PREDICTED, Names.NUM_FEATURES_USED, Names.DATA_DESCRIPTION,
                             Names.GROUPING_DESCRIPTION, Names.NORMALIZED, Names.NUM_FEATURES_NORMALIZED, Names.FEATURE_EXTRACTION]
        self.loo_summarized_classification_leaderboard_cols = summarized_cols_1 + self.mean_classification_metrics + summarized_cols_2
        self.loo_summarized_regression_leaderboard_cols = summarized_cols_1 + self.mean_regression_metrics + summarized_cols_2

        self.leaderboard_names_dict = {Names.CUSTOM_CLASS_LBOARD: self.custom_classification_leaderboard_cols,
                                       Names.CUSTOM_REG_LBOARD: self.custom_regression_leaderboard_cols,
                                       Names.LOO_SUMM_CLASS_LBOARD: self.loo_summarized_classification_leaderboard_cols,
                                       Names.LOO_SUMM_REG_LBOARD: self.loo_summarized_regression_leaderboard_cols,
                                       Names.LOO_FULL_CLASS_LBOARD: self.loo_full_classification_leaderboard_cols,
                                       Names.LOO_FULL_REG_LBOARD: self.loo_full_regression_leaderboard_cols}
        self.valid_feature_extraction_methods = [Names.ELI5_PERMUTATION,
                                                 Names.RFPIMP_PERMUTATION,
                                                 Names.BBA_AUDIT,
                                                 Names.SHAP_AUDIT]
        self.list_of_this_instance_run_ids = []
        self.dict_of_instance_run_loo_ids = defaultdict(list)
        print()

    # def train_only(self, function_that_returns_TH_model, dict_of_function_parameters, training_data,
    #                description, target_cols, feature_cols_to_use, index_cols=("dataset", "name"), normalize=False,
    #                feature_cols_to_normalize=None, feature_extraction=False, sparse_cols_to_use=None):
    #     self.run_custom(function_that_returns_TH_model=function_that_returns_TH_model,
    #                     dict_of_function_parameters=dict_of_function_parameters,
    #                     training_data=training_data, testing_data=training_data,  # both are the same for train_only
    #                     description=description,
    #                     target_cols=target_cols, feature_cols_to_use=feature_cols_to_use, index_cols=index_cols,
    #                     normalize=normalize, feature_cols_to_normalize=feature_cols_to_normalize,
    #                     feature_extraction=feature_extraction, sparse_cols_to_use=sparse_cols_to_use,
    #                     predict_untested_data=False, interpret_complex_model=False, custom_metric=False)

    def predict_only(self, run_id_of_saved_model, data_to_predict, index_cols, target_col, feature_cols_to_use):
        """
        TODO: Need to read in saved normalizations too
        TODO: sparse_cols_to_use
        TODO: potentially make an internal table that tracks prediction runs/outputs
          - for now it will always output the prediction to predicted_data.csv in the appropriate run folder.
        """
        run_id_of_saved_model = 'run_'+run_id_of_saved_model
        run_id_folder_path_of_saved_model = os.path.join(self.runs_folder_path, run_id_of_saved_model)

        run_object = _BaseRun(test_harness_model=run_id_folder_path_of_saved_model, training_data=None, testing_data=None,
                              target_col=target_col, feature_cols_to_use=feature_cols_to_use,
                              index_cols=index_cols, normalize=False, feature_cols_to_normalize=False, feature_extraction=False,
                              predict_untested_data=data_to_predict)

        # call run object methods
        start = time.time()
        print('-' * 100)  # this adds a line of dashes to signify the beginning of the model run
        print('Starting prediction_only model at time {}'.format(datetime.now().strftime("%H:%M:%S")))

        run_object.predict_only()
        self._output_run_files(run_object, run_id_folder_path_of_saved_model, True, None)
        end = time.time()
        print('Run finished at {}.'.format(datetime.now().strftime("%H:%M:%S")), 'Total run time = {0:.2f} seconds'.format(end - start))
        print('^' * 100)  # this adds a line of ^ to signify the end of of the model run
        print("\n\n\n")

    # TODO: add more normalization options: http://benalexkeen.com/feature-scaling-with-scikit-learn/
    def run_custom(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                   description, target_cols, feature_cols_to_use, index_cols=("dataset", "name"), normalize=False,
                   feature_cols_to_normalize=None, feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None,
                   interpret_complex_model=False, custom_metric=False):
        """
        Instantiates and runs a model on a custom train/test split
        If you pass in a list of columns to predict, a separate run will occur for each string in the list
        :param custom_metric: dict with string keys and values are functions that take two arguuments.  Not tested with LOO runs.
        """
        target_cols = make_list_if_not_list(target_cols)
        assert is_list_of_strings(target_cols), "target_cols must be a string or a list of strings"

        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        if feature_cols_to_normalize:
            feature_cols_to_normalize = make_list_if_not_list(feature_cols_to_normalize)
        if sparse_cols_to_use:
            sparse_cols_to_use = make_list_if_not_list(sparse_cols_to_use)
        if custom_metric:
            assert isinstance(custom_metric, dict), "custom_metric must be a dict whose key is a string and value is a function"
            self.regression_metrics.extend(list(custom_metric.keys()))
            self.custom_regression_leaderboard_cols.extend(list(custom_metric.keys()))

        for col in target_cols:
            self._execute_run(function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                              description, col, feature_cols_to_use, index_cols, normalize, feature_cols_to_normalize,
                              feature_extraction, predict_untested_data, sparse_cols_to_use, loo_dict=False,
                              interpret_complex_model=interpret_complex_model, custom_metric=custom_metric)

    def make_grouping_df(self, grouping, data):
        # if grouping is a string, turn it into a list containing that one string
        if isinstance(grouping, string_types):
            grouping = make_list_if_not_list(grouping)
        # if grouping is a list of strings:
        # 1. check if those strings exist as column names in the data Dataframe
        # 2. then create a grouping Dataframe based on the unique values in those columns
        data_cols = data.columns.values.tolist()
        if is_list_of_strings(grouping):
            # this for loop check is similar to the one for the grouping_df, but I like to have this one too for a clearer error message
            for col_name in grouping:
                assert (col_name in data_cols), \
                    "{} does not exist as a column in the data Dataframe. " \
                    "If you pass in a list of strings to the 'grouping' argument, " \
                    "then all of those strings must exist as columns in the data Dataframe.".format(col_name)
            grouping_df = data.groupby(by=grouping, as_index=False).first()[grouping]
            grouping_df[Names.GROUP_INDEX] = grouping_df.index
        elif isinstance(grouping, pd.DataFrame):
            grouping_df = grouping.copy()
        else:
            raise ValueError("grouping must be a list of column names in the data Dataframe, "
                             "or a Pandas Dataframe that defines custom groupings (see the Test Harness README for an example).")
            # TODO: add example grouping_df to README
            # grouping_df checks:
            # 1. "group_index" must exist as a column in grouping_df
            # 2. every other column in grouping_df must also be a column in the data Dataframe
        grouping_df_cols = grouping_df.columns.values.tolist()
        assert (Names.GROUP_INDEX in grouping_df_cols), "grouping_df must have a '{}' column.".format(
            Names.GROUP_INDEX)
        cols_to_group_on = [col for col in grouping_df_cols if col != Names.GROUP_INDEX]
        for col_name in cols_to_group_on:
            assert (col_name in data_cols,
                    "{} is a column in grouping_df but does not exist as a column in the data Dataframe. " \
                    "Every column in grouping_df (other than '{}') must also be a column in the data Dataframe.".format(
                        col_name,
                        Names.GROUP_INDEX))
        return grouping_df, data_cols, cols_to_group_on

    # TODO: add sparse cols to leave one out
    def run_leave_one_out(self, function_that_returns_TH_model, dict_of_function_parameters, data, data_description, grouping,
                          grouping_description, target_cols, feature_cols_to_use, index_cols=("dataset", "name"), normalize=False,
                          feature_cols_to_normalize=None, feature_extraction=False, sparse_cols_to_use=None):
        """
        Splits the data into appropriate train/test splits according to the grouping dataframe, and then runs a separate instantiation of
        the passed-in model on each split.
        """
        date_loo_ran = datetime.now().strftime("%Y-%m-%d")
        time_loo_ran = datetime.now().strftime("%H:%M:%S")

        target_cols = make_list_if_not_list(target_cols)
        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        if feature_cols_to_normalize:
            feature_cols_to_normalize = make_list_if_not_list(feature_cols_to_normalize)
            num_features_normalized = len(feature_cols_to_normalize)
        else:
            num_features_normalized = 0

        assert isinstance(data, pd.DataFrame), "data must be a Pandas Dataframe"
        assert isinstance(data_description, string_types), "data_description must be a string"
        assert isinstance(grouping_description, string_types), "grouping_description must be a string"
        assert is_list_of_strings(target_cols), "target_cols must be a string or a list of strings"

        grouping_df, data_cols, cols_to_group_on = self.make_grouping_df(grouping, data)

        # Append a "group_index" column to the all_data Dataframe. This column contains the group number of each row.
        # The values of the "group_index" column are determined from the grouping Dataframe (grouping_df)
        all_data = data.copy()
        all_data = pd.merge(left=all_data, right=grouping_df, how="left", on=cols_to_group_on)

        for col in target_cols:
            loo_id = get_id()
            loo_folder_path = os.path.join(self.runs_folder_path, '{}_{}'.format("loo", loo_id))
            os.makedirs(loo_folder_path, exist_ok=False)
            if self.compress_large_csvs:
                data.to_csv(os.path.join(loo_folder_path, "data.csv.gz"), index=False, compression="gzip")
            else:
                data.to_csv(os.path.join(loo_folder_path, "data.csv"), index=False)
            grouping_df.to_csv(os.path.join(loo_folder_path, "grouping_df.csv"), index=False)

            dummy_th_model = function_that_returns_TH_model(**dict_of_function_parameters)
            if isinstance(dummy_th_model, ClassificationModel):
                task_type = "Classification"
            elif isinstance(dummy_th_model, RegressionModel):
                task_type = "Regression"
            else:
                raise ValueError("function_that_returns_TH_model must return a ClassificationModel or a RegressionModel.")

            # iterate through the groups (determined by "group_index" column) in the all_data Dataframe:
            for i, group_index in enumerate(list(set(all_data[Names.GROUP_INDEX]))):
                description = "{}".format(data_description)
                group_rows = grouping_df.loc[grouping_df[Names.GROUP_INDEX] == group_index]
                group_info = group_rows.to_dict(orient='list')
                print("Creating test split based on {} {}".format(Names.GROUP_INDEX, group_index))
                print("example groupingdf row for the loo group: {}".format(group_rows.iloc[0]))
                if OUTPUT == Names.VERBOSE_OUTPUT:
                    print("Defined by: {}".format(group_info))
                train_split = all_data.copy()
                test_split = all_data.copy()
                train_split = train_split.loc[train_split[Names.GROUP_INDEX] != group_index]
                test_split = test_split.loc[test_split[Names.GROUP_INDEX] == group_index]

                print("Number of samples in train split:", train_split.shape)
                print("Number of samples in test split:", test_split.shape)

                loo_dict = {"loo_id": loo_id, "task_type": task_type, "data_description": data_description,
                            "grouping_description": grouping_description, "group_info": group_info}

                self._execute_run(function_that_returns_TH_model=function_that_returns_TH_model,
                                  dict_of_function_parameters=dict_of_function_parameters,
                                  training_data=train_split,
                                  testing_data=test_split,
                                  description=description,
                                  target_col=col,
                                  feature_cols_to_use=feature_cols_to_use,
                                  index_cols=index_cols,
                                  normalize=normalize,
                                  feature_cols_to_normalize=feature_cols_to_normalize,
                                  feature_extraction=feature_extraction,
                                  predict_untested_data=False,
                                  sparse_cols_to_use=sparse_cols_to_use,
                                  loo_dict=loo_dict,
                                  interpret_complex_model=False)

            # summary results are calculated here, and summary leaderboards are updated
            summary_values = {Names.LOO_ID: loo_id, Names.DATE: date_loo_ran, Names.TIME: time_loo_ran,
                              Names.MODEL_NAME: dummy_th_model.model_name, Names.MODEL_AUTHOR: dummy_th_model.model_author,
                              Names.MODEL_DESCRIPTION: dummy_th_model.model_description, Names.COLUMN_PREDICTED: col,
                              Names.NUM_FEATURES_USED: len(feature_cols_to_use), Names.DATA_DESCRIPTION: data_description,
                              Names.GROUPING_DESCRIPTION: grouping_description, Names.NORMALIZED: normalize,
                              Names.NUM_FEATURES_NORMALIZED: num_features_normalized, Names.FEATURE_EXTRACTION: feature_extraction}
            if task_type == "Classification":
                self.output_classification_leaderboard_to_csv(summary_values, loo_id)
            elif task_type == "Regression":
                self.output_regression_leaderboard_to_csv(summary_values, loo_id)
            else:
                raise TypeError("task_type must be 'Classification' or 'Regression'.")

    def output_classification_leaderboard_to_csv(self, summary_values, loo_id):
        detailed_leaderboard_name = Names.LOO_FULL_CLASS_LBOARD
        detailed_leaderboard_path = os.path.join(self.results_folder_path, "{}.html".format(detailed_leaderboard_name))
        detailed_leaderboard = pd.read_html(detailed_leaderboard_path)[0]
        this_loo_results = detailed_leaderboard.loc[detailed_leaderboard[Names.LOO_ID] == loo_id]

        summary_metrics = {}
        for metric, mean_metric in zip(self.classification_metrics, self.mean_classification_metrics):
            summary_metrics[mean_metric] = mean(this_loo_results[metric])
            # TODO: add standard deviation with pstdev
        summary_values.update(summary_metrics)

        # Update summary leaderboard
        summary_leaderboard_name = Names.LOO_SUMM_CLASS_LBOARD
        summary_leaderboard_cols = self.loo_summarized_classification_leaderboard_cols
        # first check if leaderboard exists and create empty leaderboard if it doesn't
        html_path = os.path.join(self.results_folder_path, "{}.html".format(summary_leaderboard_name))
        try:
            summary_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            summary_leaderboard = pd.DataFrame(columns=summary_leaderboard_cols)

        # update leaderboard with new entry (row_of_results) and sort it based on run type
        summary_leaderboard = summary_leaderboard.append(summary_values, ignore_index=True, sort=False)
        sort_metric = "Mean " + self.metric_to_sort_classification_results_by
        summary_leaderboard.sort_values(sort_metric, inplace=True, ascending=False)
        summary_leaderboard.reset_index(inplace=True, drop=True)

        # overwrite old leaderboard with updated leaderboard
        summary_leaderboard.to_html(html_path, index=False, classes=summary_leaderboard_name)
        if self.output_csvs_of_leaderboards is True:
            csv_path = os.path.join(self.results_folder_path, "{}.csv".format(summary_leaderboard_name))
            summary_leaderboard.to_csv(csv_path, index=False)

    def output_regression_leaderboard_to_csv(self, summary_values, loo_id):
        detailed_leaderboard_name = Names.LOO_FULL_REG_LBOARD
        detailed_leaderboard_path = os.path.join(self.results_folder_path, "{}.html".format(detailed_leaderboard_name))
        detailed_leaderboard = pd.read_html(detailed_leaderboard_path)[0]
        this_loo_results = detailed_leaderboard.loc[detailed_leaderboard[Names.LOO_ID] == loo_id]

        summary_metrics = {}
        for metric, mean_metric in zip(self.regression_metrics, self.mean_regression_metrics):
            summary_metrics[mean_metric] = mean(this_loo_results[metric])
            # TODO: add standard deviation with pstdev
        summary_values.update(summary_metrics)

        # Update summary leaderboard
        summary_leaderboard_name = Names.LOO_SUMM_REG_LBOARD
        summary_leaderboard_cols = self.loo_summarized_regression_leaderboard_cols
        # first check if leaderboard exists and create empty leaderboard if it doesn't
        html_path = os.path.join(self.results_folder_path, "{}.html".format(summary_leaderboard_name))
        try:
            summary_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            summary_leaderboard = pd.DataFrame(columns=summary_leaderboard_cols)

        # update leaderboard with new entry (row_of_results) and sort it based on run type
        summary_leaderboard = summary_leaderboard.append(summary_values, ignore_index=True, sort=False)
        sort_metric = "Mean " + self.metric_to_sort_regression_results_by
        print("Leave-One-Out Summary Leaderboard:\n")
        print(summary_leaderboard)
        summary_leaderboard.sort_values(sort_metric, inplace=True, ascending=False)
        summary_leaderboard.reset_index(inplace=True, drop=True)

        # overwrite old leaderboard with updated leaderboard
        summary_leaderboard.to_html(html_path, index=False, classes=summary_leaderboard_name)
        if self.output_csvs_of_leaderboards is True:
            csv_path = os.path.join(self.results_folder_path, "{}.csv".format(summary_leaderboard_name))
            summary_leaderboard.to_csv(csv_path, index=False)

    def validate_execute_run_inputs(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                                    description, target_col, feature_cols_to_use, index_cols, normalize,
                                    feature_cols_to_normalize, feature_extraction, predict_untested_data, sparse_cols_to_use,
                                    custom_metric):
        # Single strings are included in the assert error messages because the make_list_if_not_list function was used
        assert callable(function_that_returns_TH_model), \
            "function_that_returns_TH_model must be a function that returns a TestHarnessModel object"
        assert isinstance(dict_of_function_parameters, dict), \
            "dict_of_function_parameters must be a dictionary of parameters for the function_that_returns_TH_model function."
        assert isinstance(training_data, pd.DataFrame), "training_data must be a Pandas Dataframe"
        assert isinstance(testing_data, pd.DataFrame), "testing_data must be a Pandas Dataframe"
        assert isinstance(description, string_types), "description must be a string"
        assert isinstance(target_col, string_types), "target_col must be a string"
        assert is_list_of_strings(feature_cols_to_use), "feature_cols_to_use must be a string or a list of strings"
        assert isinstance(normalize, bool), "normalize must be True or False"
        assert (feature_cols_to_normalize is None) or is_list_of_strings(feature_cols_to_normalize), \
            "feature_cols_to_normalize must be None, a string, or a list of strings"
        assert isinstance(feature_extraction, bool) or (feature_extraction in self.valid_feature_extraction_methods), \
            "feature_extraction must be a bool or one of the following strings: {}".format(self.valid_feature_extraction_methods)
        assert (predict_untested_data is False) or (isinstance(predict_untested_data, pd.DataFrame)), \
            "predict_untested_data must be False or a Pandas Dataframe"
        assert (sparse_cols_to_use is None) or is_list_of_strings(sparse_cols_to_use), \
            "sparse_cols_to_use must be None, a string, or a list of strings"
        assert (index_cols is None) or (isinstance(index_cols, list)) or (isinstance(index_cols, tuple)), \
            "index_cols must be None or a list (or tuple) of index column names in the passed-in training, testing, and prediction data."
        if isinstance(index_cols, tuple):
            index_cols = list(index_cols)
        if isinstance(index_cols, list):
            assert is_list_of_strings(index_cols), "if index_cols is a tuple or list, it must contain only strings."
        if custom_metric:
            assert type(
                custom_metric) is dict, 'Custom metric must be of type dict. Key should be string, and value should a be a function that takes in two arguuments.'

        # check if index_cols exist in training, testing, and prediction dataframes:
        assert (set(index_cols).issubset(training_data.columns.tolist())), \
            "the strings in index_cols are not valid columns in training_data."
        assert (set(index_cols).issubset(testing_data.columns.tolist())), \
            "the strings in index_cols are not valid columns in testing_data."
        if isinstance(predict_untested_data, pd.DataFrame):
            assert (set(index_cols).issubset(predict_untested_data.columns.tolist())), \
                "the strings in index_cols are not valid columns in predict_untested_data."

    # TODO: replace loo_dict with type_dict --> first entry is run type --> this will allow for more types in the future
    def _execute_run(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                     description, target_col, feature_cols_to_use, index_cols=("dataset", "name"), normalize=False,
                     feature_cols_to_normalize=None, feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None,
                     loo_dict=False, interpret_complex_model=False, custom_metric=False):
        """
        1. Instantiates the TestHarnessModel object
        2. Creates a _BaseRun object and calls their train_and_test_model and calculate_metrics methods
        3. Calls _output_results(Run Object)
        """
        # TODO: add checks to ensure index_cols represent unique values in training, testing, and prediction dataframes
        self.validate_execute_run_inputs(function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                                         description, target_col, feature_cols_to_use, index_cols, normalize,
                                         feature_cols_to_normalize, feature_extraction, predict_untested_data, sparse_cols_to_use,
                                         custom_metric)

        train_df, test_df = training_data.copy(), testing_data.copy()
        if isinstance(predict_untested_data, pd.DataFrame):
            pred_df = predict_untested_data.copy()
        else:
            pred_df = False

        # for each col in index_cols, create a copy with and "unchanged_" prefix added, because later we want to
        # output the original column that hasn't been changed by operations such as normalization
        for col in index_cols:
            train_df["unchanged_{}".format(col)] = train_df[col]
            test_df["unchanged_{}".format(col)] = test_df[col]
            if isinstance(pred_df, pd.DataFrame):
                pred_df["unchanged_{}".format(col)] = pred_df[col]

        test_harness_model = function_that_returns_TH_model(**dict_of_function_parameters)

        # This is the one and only time _BaseRun is invoked
        run_object = _BaseRun(test_harness_model, train_df, test_df, description, target_col,
                              copy(feature_cols_to_use), copy(index_cols), normalize, copy(feature_cols_to_normalize), feature_extraction,
                              pred_df, copy(sparse_cols_to_use), loo_dict, interpret_complex_model, custom_metric)

        # tracking the run_ids of all the runs that were kicked off in this TestHarness instance
        loo_id = None
        if loo_dict:
            loo_id = run_object.loo_dict.get('loo_id')
        if loo_id is not None:
            self.dict_of_instance_run_loo_ids[loo_id].append(run_object.run_id)
        else:
            self.list_of_this_instance_run_ids.append(run_object.run_id)

        # call run object methods
        start = time.time()
        # this adds a line of dashes to signify the beginning of the model run
        print('-' * 100)

        print('Starting run of model {} at time {}'.format(datetime.now().strftime("%H:%M:%S"), function_that_returns_TH_model.__name__))
        run_object.train_and_test_model()
        run_object.calculate_metrics()

        if run_object.feature_extraction is not False:
            from harness.feature_extraction import FeatureExtractor
            feature_extractor = FeatureExtractor(base_run_instance=run_object)
            feature_extractor.feature_extraction_method(method=run_object.feature_extraction)
        else:
            feature_extractor = None

        # ----------------------------------
        # model on model
        if interpret_complex_model:
            run_object.interpret_model(
                complex_model=run_object.test_harness_model.model,
                training_df=run_object.training_data,
                feature_col=run_object.feature_cols_to_use,
                predict_col=run_object.target_col,
                simple_model=None)
        # ----------------------------------

        # output results of run object by updating the appropriate leaderboard(s) and writing files to disk
        # Pandas append docs: "Columns not in this frame are added as new columns" --> don't worry about adding new leaderboard cols

        self._update_leaderboard(run_object)

        if run_object.loo_dict is False:
            run_id_folder_path = os.path.join(self.runs_folder_path, '{}_{}'.format("run", run_object.run_id))
            os.makedirs(run_id_folder_path)
            self._output_run_files(run_object, run_id_folder_path, True, feature_extractor)
        else:
            loo_id = run_object.loo_dict['loo_id']
            loo_path = os.path.join(self.runs_folder_path, '{}_{}'.format("loo", loo_id))
            os.makedirs(loo_path, exist_ok=True)
            run_id_folder_path = os.path.join(loo_path, '{}_{}'.format("run", run_object.run_id))
            os.makedirs(run_id_folder_path)
            self._output_run_files(run_object, run_id_folder_path, True, feature_extractor)

        end = time.time()
        print('Run finished at {}.'.format(datetime.now().strftime("%H:%M:%S")), 'Total run time = {0:.2f} seconds'.format(end - start))
        print('^' * 100)  # this adds a line of ^ to signify the end of of the model run
        print("\n\n\n")

    def _update_leaderboard(self, run_object):
        # find appropriate leaderboard to update based on run_object characteristics
        if run_object.loo_dict is False:
            if run_object.run_type == Names.CLASSIFICATION:
                leaderboard_name = Names.CUSTOM_CLASS_LBOARD
            elif run_object.run_type == Names.REGRESSION:
                leaderboard_name = Names.CUSTOM_REG_LBOARD
            else:
                raise TypeError("run_object.run_type must equal '{}' or '{}'".format(Names.CLASSIFICATION, Names.REGRESSION))
        else:
            if run_object.run_type == Names.CLASSIFICATION:
                leaderboard_name = Names.LOO_FULL_CLASS_LBOARD
            elif run_object.run_type == Names.REGRESSION:
                leaderboard_name = Names.LOO_FULL_REG_LBOARD
            else:
                raise TypeError("run_object.run_type must equal '{}' or '{}'".format(Names.CLASSIFICATION, Names.REGRESSION))
        assert leaderboard_name in self.leaderboard_names_dict.keys(), "passed-in leaderboard_name is not valid."
        leaderboard_cols = self.leaderboard_names_dict[leaderboard_name]

        # first check if leaderboard exists and create empty leaderboard if it doesn't
        html_path = os.path.join(self.results_folder_path, "{}.html".format(leaderboard_name))
        try:
            leaderboard = pd.read_html(html_path)[0]

        except (IOError, ValueError):
            leaderboard = pd.DataFrame(columns=leaderboard_cols)

        # create leaderboard entry for this run and add two LOO-specific columns if loo_dict exists
        row_of_results = self._create_row_entry(run_object)
        if run_object.loo_dict is not False:
            row_of_results[Names.LOO_ID] = run_object.loo_dict["loo_id"]
            row_of_results[Names.TEST_GROUP] = str(run_object.loo_dict["group_info"])
        if OUTPUT == Names.VERBOSE_OUTPUT:
            print()
            print(row_of_results)
            print()

        # update leaderboard with new entry (row_of_results) and sort it based on run type
        leaderboard = leaderboard.append(row_of_results, ignore_index=True, sort=False)  # sort=False prevents columns from reordering

        # If the custom metric is changed or removed,
        # then make sure you put NaN in the slot that you had before so that you don't lose that column

        if len(set(leaderboard.columns).symmetric_difference(row_of_results.columns)) > 0:
            cols = set(leaderboard.columns).symmetric_difference(row_of_results.columns)
            for col in cols:
                row_of_results[col] = 'NaN'

        leaderboard = leaderboard.reindex(row_of_results.columns, axis=1)  # reindex will correct col order in case a new col is added
        if run_object.run_type == Names.CLASSIFICATION:
            leaderboard.sort_values(self.metric_to_sort_classification_results_by, inplace=True, ascending=False)
        elif run_object.run_type == Names.REGRESSION:
            # print(leaderboard[self.metric_to_sort_regression_results_by].value_counts(dropna=False))
            leaderboard.sort_values(self.metric_to_sort_regression_results_by, inplace=True, ascending=False)
        else:
            raise TypeError("run_object.run_type must equal '{}' or '{}'".format(Names.CLASSIFICATION, Names.REGRESSION))
        leaderboard.reset_index(inplace=True, drop=True)

        # overwrite old leaderboard with updated leaderboard
        leaderboard.to_html(html_path, index=False, classes=leaderboard_name)
        if self.output_csvs_of_leaderboards is True:
            csv_path = os.path.join(self.results_folder_path, "{}.csv".format(leaderboard_name))
            leaderboard.to_csv(csv_path, index=False)

    def _create_row_entry(self, run_object):

        row_values = {Names.RUN_ID: run_object.run_id, Names.DATE: run_object.date_ran, Names.TIME: run_object.time_ran,
                      Names.SAMPLES_IN_TRAIN: run_object.metrics_dict[Names.SAMPLES_IN_TRAIN],
                      Names.SAMPLES_IN_TEST: run_object.metrics_dict[Names.SAMPLES_IN_TEST],
                      Names.MODEL_NAME: run_object.model_name, Names.MODEL_AUTHOR: run_object.model_author,
                      Names.MODEL_DESCRIPTION: run_object.model_description, Names.COLUMN_PREDICTED: run_object.target_col,
                      Names.NUM_FEATURES_USED: run_object.metrics_dict[Names.NUM_FEATURES_USED],
                      Names.DESCRIPTION: run_object.description, Names.NORMALIZED: run_object.normalize,
                      Names.NUM_FEATURES_NORMALIZED: run_object.metrics_dict[Names.NUM_FEATURES_NORMALIZED],
                      Names.FEATURE_EXTRACTION: run_object.feature_extraction,
                      Names.WAS_UNTESTED_PREDICTED: run_object.was_untested_data_predicted}
        if run_object.run_type == Names.CLASSIFICATION:
            # extract relevant metrics from run_object.metrics_dict and round to 3rd decimal place:
            metric_results = {metric: round(run_object.metrics_dict[metric], 3) for metric in self.classification_metrics}
            row_values.update(metric_results)
            row_of_results = pd.DataFrame(columns=self.custom_classification_leaderboard_cols)
            row_of_results = row_of_results.append(row_values, ignore_index=True, sort=False)
        elif run_object.run_type == Names.REGRESSION:
            # extract relevant metrics from run_object.metrics_dict and round to 3rd decimal place:
            metric_results = {metric: round(run_object.metrics_dict[metric], 3) for metric in self.regression_metrics}
            row_values.update(metric_results)
            row_of_results = pd.DataFrame(columns=self.custom_regression_leaderboard_cols)
            row_of_results = row_of_results.append(row_values, ignore_index=True, sort=False)
        else:
            raise ValueError("run_object.run_type must be {} or {}".format(Names.REGRESSION, Names.CLASSIFICATION))
        return row_of_results

    def _output_run_files(self, run_object, output_path, output_data_csvs=True, feature_extractor=None, output_model=True):
        if output_data_csvs:
            # using index_cols and prediction/ranking cols to only output subset of dataframe.
            # using unchanged_index_cols to get names of columns that were created in execute_run for later output.
            # thus what is output are the original input columns and not transformed input columns (e.g. if normalization is used)

            unchanged_index_cols = ["unchanged_{}".format(x) for x in run_object.index_cols]

            # creating list of cols to output for train, test, and pred outputs
            train_cols_to_output = unchanged_index_cols + [run_object.target_col]
            if run_object.run_type == Names.CLASSIFICATION:
                test_cols_to_output = train_cols_to_output + [run_object.predictions_col, run_object.prob_predictions_col]
                pred_cols_to_output = unchanged_index_cols + [run_object.predictions_col, run_object.prob_predictions_col,
                                                              run_object.rankings_col]
            elif run_object.run_type == Names.REGRESSION:
                test_cols_to_output = unchanged_index_cols + [run_object.predictions_col, run_object.residuals_col]
                pred_cols_to_output = unchanged_index_cols + [run_object.predictions_col, run_object.rankings_col]
            elif run_object.run_type == Names.PREDICT_ONLY:
                test_cols_to_output = None
                pred_cols_to_output = unchanged_index_cols + [run_object.predictions_col, run_object.rankings_col]
            else:
                raise ValueError("run_object.run_type must be {} or {}".format(Names.REGRESSION, Names.CLASSIFICATION, Names.PREDICT_ONLY))

            if run_object.run_type != Names.PREDICT_ONLY:
                train_df_to_output = run_object.training_data[train_cols_to_output].copy()
                for col in unchanged_index_cols:
                    train_df_to_output.rename(columns={col: col.rsplit("unchanged_")[1]}, inplace=True)
                test_df_to_output = run_object.testing_data_predictions[test_cols_to_output].copy()
                for col in unchanged_index_cols:
                    test_df_to_output.rename(columns={col: col.rsplit("unchanged_")[1]}, inplace=True)
                if self.compress_large_csvs:
                    train_df_to_output.to_csv('{}/{}'.format(output_path, 'training_data.csv.gz'), index=False, compression="gzip")
                    test_df_to_output.to_csv('{}/{}'.format(output_path, 'testing_data.csv.gz'), index=False, compression="gzip")
                else:
                    train_df_to_output.to_csv('{}/{}'.format(output_path, 'training_data.csv'), index=False)
                    test_df_to_output.to_csv('{}/{}'.format(output_path, 'testing_data.csv'), index=False)

            if run_object.was_untested_data_predicted is not False:
                # TODO: make this work, using simpler output for now:
                '''
                prediction_data_to_output = run_object.untested_data_predictions[pred_cols_to_output].copy()
                for col in unchanged_index_cols:
                    prediction_data_to_output.rename(columns={col: col.rsplit("unchanged_")[1]}, inplace=True)
                prediction_data_to_output.to_csv('{}/{}'.format(output_path, 'predicted_data.csv'), index=False)
                '''
                prediction_data_to_output = run_object.untested_data_predictions.copy()
                if self.compress_large_csvs:
                    prediction_data_to_output.to_csv('{}/{}'.format(output_path, 'predicted_data.csv.gz'), index=False, compression="gzip")
                else:
                    prediction_data_to_output.to_csv('{}/{}'.format(output_path, 'predicted_data.csv'), index=False)

        if run_object.feature_extraction is not False:
            from harness.feature_extraction import FeatureExtractor
            assert isinstance(feature_extractor, FeatureExtractor), \
                "feature_extractor must be a FeatureExtractor object when run_object.feature_extraction is not False."
            feature_extractor.feature_importances.to_csv('{}/{}'.format(output_path, 'feature_importances.csv'), index=False)
            if run_object.feature_extraction == Names.SHAP_AUDIT:
                shap_path = os.path.join(output_path, 'SHAP')
                if not os.path.exists(shap_path):
                    os.makedirs(shap_path)
                dependence_path = os.path.join(shap_path, 'feature_dependence_plots')
                if not os.path.exists(dependence_path):
                    os.makedirs(dependence_path)
                # feature_extractor.shap_values.to_csv('{}/{}'.format(shap_path, 'shap_values.csv'), index=False)
                for name, plot in feature_extractor.shap_plots_dict.items():
                    if "dependence_plot" in name:
                        plot.savefig(os.path.join(dependence_path, name), bbox_inches="tight")
                    else:
                        plot.savefig(os.path.join(shap_path, name), bbox_inches="tight")

            if run_object.feature_extraction == Names.BBA_AUDIT:
                bba_path = os.path.join(output_path, 'BBA')
                if not os.path.exists(bba_path):
                    os.makedirs(bba_path)
                for name, plot in feature_extractor.bba_plots_dict.items():
                    plot.savefig(os.path.join(bba_path, name), bbox_inches="tight")

        # model on model 
        if run_object.interpret_complex_model is True:
            import pydotplus

            img_string_path = os.path.join(output_path, 'Complex_Model_Interpretation')
            if not os.path.exists(img_string_path):
                os.makedirs(img_string_path)
            img_string = run_object.model_interpretation_img.getvalue()
            with open(os.path.join(img_string_path, 'model_interpretation_string.txt'), 'w') as f:
                f.write(img_string)
                f.close()

            image_path = os.path.join(output_path, 'Complex_Model_Interpretation')
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            img = pydotplus.graph_from_dot_data(run_object.model_interpretation_img.getvalue())
            img.write_png(os.path.join(image_path, 'model_interpretation.png'))

        if run_object.run_type != Names.PREDICT_ONLY:
            test_file_name = os.path.join(output_path, 'model_information.txt')
            with open(test_file_name, "w") as f:
                f.write("%s\n" % run_object.model_name)
                f.write("Feature columns used by model: \n")
                json.dump(run_object.feature_cols_to_use, f)
                f.write("\n\n\n")

                f.write("Model Instantiation Trace:\n")
                for i, t in enumerate(run_object.model_stack_trace):
                    f.write(" Level {}\n".format(i))
                    path, line, func = t[1:4]
                    f.write(' - Path: ' + path + '\n')
                    f.write(' - Line: ' + str(line) + ',  Function: ' + str(func) + '\n')
                    f.write("\n")

            if run_object.normalization_scaler_object is not None:
                joblib.dump(run_object.normalization_scaler_object, os.path.join(output_path, "normalization_scaler_object.pkl"))

            if output_model:
                joblib.dump(run_object.test_harness_model.model, os.path.join(output_path, "trained_model.pkl"))

    def print_leaderboards(self):
        pass
