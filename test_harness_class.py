import os
import json
import time
import itertools
import pandas as pd
from unique_id import get_id
from six import string_types
from datetime import datetime
from statistics import mean, pstdev
from run_classes import ClassificationRun, RegressionRun
from test_harness.test_harness_models_abstract_classes import TestHarnessModel, ClassificationModel, RegressionModel

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
# CSS classes applied to the Pandas Dataframes when written as HTML
css_classes = ["table-bordered", "table-striped", "table-compact"]

PWD = os.getcwd()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)
DEFAULT_DATA_PATH = os.path.join(PWD, 'versioned_data/asap/')


# TODO: If model, training_data, and other params are the same, just train once for that call of run_models
# TODO: add ran-by (user) column to leaderboards
# TODO: add md5hashes of data to leaderboard as sorting tool
# TODO: add cross validation
# TODO: if test set doesn't include col_to_predict, carry out prediction instead?
# TODO: add more checks for correct inputs using assert
# TODO: add filelock or writing-scheduler so leaderboards are not overwritten at the same time. Might need to use SQL
# TODO: by having the ability to "add" multiple models to the TestHarness object, you can allow for visualizations or \
# TODO: summary stats for a certain group of runs by adding arguments to the execute_runs method!
# TODO: potentially move sparse column code to run_classes.py so LOO can use it as well easily


def make_list_if_not_list(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


def is_list_of_strings(obj):
    if obj and isinstance(obj, list):
        return all(isinstance(elem, string_types) for elem in obj)
    else:
        return False


# TODO: separate data description from split description
class TestHarness:
    def __init__(self, output_path=os.path.dirname(os.path.realpath(__file__))):
        # Note: loo stands for leave-one-out
        self.output_path = output_path
        self.results_folder_path = os.path.join(self.output_path, 'results')
        self._runs_to_execute = []
        self._execution_id = None
        self._finished_custom_runs = []
        self._finished_loo_runs = {}
        self._flag_to_only_allow_one_execution_of_runs_per_TestHarness_object = False
        self.custom_classification_leaderboard_cols = \
            ['Execution ID', 'Run ID', 'Date', 'Time', 'AUC Score', 'Classification Accuracy', 'Model Description', 'Column Predicted',
             'Number Of Features Used', 'Data and Split Description', 'Normalized', 'Number of Features Normalized',
             'Feature Extraction', "Was Untested Data Predicted"]
        self.custom_regression_leaderboard_cols = \
            ['Execution ID', 'Run ID', 'Date', 'Time', 'R-Squared', 'RMSE', 'Model Description', 'Column Predicted',
             'Number Of Features Used', 'Data and Split Description', 'Normalized', 'Number of Features Normalized',
             'Feature Extraction', "Was Untested Data Predicted"]
        self.loo_full_classification_leaderboard_cols = \
            ['Execution ID', 'Leave-One-Out ID', 'Run ID', 'Date', 'Time', 'AUC Score', 'Classification Accuracy', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Test Group', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.loo_full_regression_leaderboard_cols = \
            ['Execution ID', 'Leave-One-Out ID', 'Run ID', 'Date', 'Time', 'R-Squared', 'RMSE', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Test Group', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.loo_summarized_classification_leaderboard_cols = \
            ['Execution ID', 'Leave-One-Out ID', 'Date', 'Time', 'Mean AUC Score', 'Mean Classification Accuracy', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Data Description', 'Grouping Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.loo_summarized_regression_leaderboard_cols = \
            ['Execution ID', 'Leave-One-Out ID', 'Date', 'Time', 'Mean R-Squared', 'Mean RMSE', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Data Description', 'Grouping Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.valid_feature_extraction_methods = ['eli5_permutation', 'rfpimp_permutation']

    # TODO: add more normalization options: http://benalexkeen.com/feature-scaling-with-scikit-learn/
    # TODO: make feature_extraction options something like: "BBA", "permutation", and "custom", where custom means that
    # TODO: it's not a black box feature tool, but rather a specific one defined inside of the TestHarnessModel object
    def add_custom_runs(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                        data_and_split_description, cols_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                        feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None):
        # Adds custom run(s) to the TestHarness object
        # If you pass a list of models and/or list of columns to predict, a custom run will be added for every
        # combination of models and columns to predict that you provided.
        # Custom runs require providing:
        #       - a function_that_returns_TH_model along with a dict_of_function_parameters for that function
        #       - a training dataframe and a testing dataframe
        #       - a column to predict or list of columns to predict
        #       - other arguments

        cols_to_predict = make_list_if_not_list(cols_to_predict)
        assert is_list_of_strings(cols_to_predict), "cols_to_predict must be a string or a list of strings"

        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        feature_cols_to_normalize = make_list_if_not_list(feature_cols_to_normalize)
        if sparse_cols_to_use is not None:
            sparse_cols_to_use = make_list_if_not_list(sparse_cols_to_use)

        for col in cols_to_predict:
            self._add_run(function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                          data_and_split_description, col, feature_cols_to_use, normalize, feature_cols_to_normalize,
                          feature_extraction, predict_untested_data, sparse_cols_to_use)

    def add_leave_one_out_runs(self, function_that_returns_TH_model, dict_of_function_parameters, data, data_description, grouping,
                               grouping_description, cols_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                               feature_extraction=False):
        # Adds leave-one-out run(s) to the TestHarness object
        # Leave-one-out runs require providing:
        #       - a function_that_returns_TH_model along with a dict_of_function_parameters for that function
        #       - a dataset dataframe
        #       - a grouping dataframe or a list of column names to group by
        #       - a column to predict or list of columns to predict
        #       - other arguments
        cols_to_predict = make_list_if_not_list(cols_to_predict)
        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        feature_cols_to_normalize = make_list_if_not_list(feature_cols_to_normalize)

        assert isinstance(data, pd.DataFrame), "data must be a Pandas Dataframe"
        assert isinstance(data_description, string_types), "data_description must be a string"
        assert isinstance(grouping, pd.DataFrame) or is_list_of_strings(
            grouping), "grouping must be a Pandas Dataframe or a list of column names"
        assert isinstance(grouping_description, string_types), "grouping_description must be a string"
        assert is_list_of_strings(cols_to_predict), "cols_to_predict must be a string or a list of strings"

        grouping = grouping.rename(columns={'name': 'topology'})
        all_data = data.copy()
        relevant_groupings = grouping.copy()
        relevant_groupings = relevant_groupings.loc[(relevant_groupings['dataset'].isin(all_data['dataset'])) &
                                                    (relevant_groupings['topology'].isin(all_data['topology']))]

        for col in cols_to_predict:
            loo_id = get_id()
            self._finished_loo_runs[loo_id] = []
            for group in list(set(relevant_groupings['group_index'])):
                data_and_split_description = "{}. Index of left-out test group = {}".format(data_description, group)
                train_split = all_data.copy()
                test_split = all_data.copy()
                print("Creating test split based on group {}:".format(group))
                group_df = relevant_groupings.loc[relevant_groupings['group_index'] == group]
                print(group_df.to_string(index=False))
                group_info = str(list(set(group_df['dataset'])) + list(set(group_df['topology'])))
                train_split = train_split.loc[~((train_split['dataset'].isin(group_df['dataset'])) &
                                                (train_split['topology'].isin(group_df['topology'])))]
                test_split = test_split.loc[(test_split['dataset'].isin(group_df['dataset'])) &
                                            (test_split['topology'].isin(group_df['topology']))]

                print("Number of samples in train split:", train_split.shape)
                print("Number of samples in test split:", test_split.shape)
                print()

                loo_dict = {"loo_id": loo_id, "data": data, "data_description": data_description, "grouping": grouping,
                            "grouping_description": grouping_description, "group_info": group_info}

                self._add_run(function_that_returns_TH_model, dict_of_function_parameters, train_split, test_split,
                              data_and_split_description, col, feature_cols_to_use, normalize, feature_cols_to_normalize,
                              feature_extraction, False, None, loo_dict)

    def _add_run(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                 data_and_split_description, col_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                 feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None, loo_dict=False):
        # Instantiates the TestHarnessModel object, Creates a ClassificationRun or RegressionRun object, and updates self._runs_to_execute
        # with the new object.

        # Single strings are included in the assert error messages because the make_list_if_not_list function was used
        assert callable(function_that_returns_TH_model), \
            "function_that_returns_TH_model must be a function that returns a TestHarnessModel object"
        assert isinstance(dict_of_function_parameters, dict), \
            "dict_of_function_parameters must be a dictionary of parameters for the function_that_returns_TH_model function."
        assert isinstance(training_data, pd.DataFrame), "training_data must be a Pandas Dataframe"
        assert isinstance(testing_data, pd.DataFrame), "testing_data must be a Pandas Dataframe"
        assert isinstance(data_and_split_description, string_types), "data_and_split_description must be a string"
        assert isinstance(col_to_predict, string_types), "col_to_predict must be a string"
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

        train_df, test_df = training_data.copy(), testing_data.copy()
        # TODO sparse_cols for untested data
        if sparse_cols_to_use is not None:
            train_df, feature_cols_to_use = self._make_sparse_cols(train_df, sparse_cols_to_use, feature_cols_to_use)
            test_df = self._make_sparse_cols(test_df, sparse_cols_to_use)

        test_harness_model = function_that_returns_TH_model(**dict_of_function_parameters)
        if isinstance(test_harness_model, ClassificationModel):
            run = ClassificationRun(test_harness_model, train_df, test_df, data_and_split_description, col_to_predict,
                                    feature_cols_to_use, normalize, feature_cols_to_normalize, feature_extraction, predict_untested_data,
                                    loo_dict)
        elif isinstance(test_harness_model, RegressionModel):
            run = RegressionRun(test_harness_model, train_df, test_df, data_and_split_description, col_to_predict,
                                feature_cols_to_use, normalize, feature_cols_to_normalize, feature_extraction, predict_untested_data,
                                loo_dict)
        else:
            raise TypeError("test_harness_model must be a ClassificationModel or a RegressionModel.")
        self._runs_to_execute.append(run)

    # Executes runs that have been added to self._custom_runs_to_execute and self._loo_runs_to_execute
    # by using _execute_custom_run and _execute_leave_one_out_run
    def execute_runs(self):
        self._execution_id = get_id()
        self.execution_id_folder_path = os.path.join(self.results_folder_path, 'executions/{}_{}'.format('exec', self._execution_id))
        print()
        print("The ID for this Execution of runs is: {}".format(self._execution_id))
        print()

        # TODO: figure out how to prevent simultaneous leaderboard updates from overwriting each other
        number_of_runs = len(self._runs_to_execute)
        if number_of_runs > 0:
            print("Executing {} runs".format(number_of_runs))
            print()
            for counter, run_object in enumerate(self._runs_to_execute, start=1):
                start = time.time()
                print('Starting run {}/{} at time {}'.format(counter, number_of_runs, datetime.now().strftime("%H:%M:%S")))

                run_object.train_and_test_model()
                run_object.calculate_metrics()
                if run_object.feature_extraction is not False:
                    run_object.feature_extraction_method(method=run_object.feature_extraction)
                if run_object.loo_dict is False:
                    self._finished_custom_runs.append(run_object)
                else:
                    loo_id = run_object.loo_dict['loo_id']
                    self._finished_loo_runs[loo_id].append(run_object)

                end = time.time()
                print('Run finished at {}'.format(datetime.now().strftime("%H:%M:%S")),
                      'Total run time = {0:.2f} seconds'.format(end - start))
                print()
            print("Outputting results from all runs...")
            self._output_results()

    # If there are categorical columns that need to be made sparse, make them, and update the feature_cols_to_use
    def _make_sparse_cols(self, df, sparse_col_names, feature_cols_to_use=None):
        """
        Take in a dataframe with the name of the columns that require construction of sparse columns.
        Update the feature columns to use list with the new sparse column construction.
        Drop the column that was made sparse from the list of features to use.
        :param df: input dataframe
        :param sparse_col_names: names of columns that need to be made sparse
        :param feature_cols_to_use: list of all features to use that needs to be updated
        :return: updated dataframe and feature columns to use
        """
        for col, col_data in df.iteritems():
            if str(col) in sparse_col_names:
                col_data = pd.get_dummies(col_data, prefix=col)
                df = df.join(col_data)
                if feature_cols_to_use:
                    feature_cols_to_use.remove(col)
                    feature_cols_to_use.extend([col for col in col_data.columns])
        if feature_cols_to_use:
            return df, feature_cols_to_use
        else:
            return df

        # TODO: Put in a check to never normalize the sparse data category

    def _output_results(self):
        custom_classification_results = pd.DataFrame(columns=self.custom_classification_leaderboard_cols)
        custom_regression_results = pd.DataFrame(columns=self.custom_regression_leaderboard_cols)

        loo_classification_results = pd.DataFrame(columns=self.loo_full_classification_leaderboard_cols)
        loo_regression_results = pd.DataFrame(columns=self.loo_full_regression_leaderboard_cols)
        loo_classification_summaries = pd.DataFrame(columns=self.loo_summarized_classification_leaderboard_cols)
        loo_regression_summaries = pd.DataFrame(columns=self.loo_summarized_regression_leaderboard_cols)

        for run_object in self._finished_custom_runs:
            run_id_folder_path = os.path.join(self.execution_id_folder_path, '{}_{}'.format("run", run_object.run_id))
            os.makedirs(run_id_folder_path, exist_ok=True)

            finished_run_result = self._output_single_run(run_object, run_id_folder_path)

            if isinstance(run_object, ClassificationRun):
                custom_classification_results = custom_classification_results.append(finished_run_result, ignore_index=True)
            elif isinstance(run_object, RegressionRun):
                custom_regression_results = custom_regression_results.append(finished_run_result, ignore_index=True)
            else:
                raise TypeError("run_object must be a ClassificationRun or RegressionRun object.")

        for this_loo_id, runs_in_this_loo in self._finished_loo_runs.items():
            # we can use the first run to extract run type and other attributes that are common across the runs in a single LOO job
            first_run_in_this_loo = runs_in_this_loo[0]
            if isinstance(first_run_in_this_loo, ClassificationRun):
                loo_run_type = "classification"
                this_loo_results = pd.DataFrame(columns=self.loo_full_classification_leaderboard_cols)
            elif isinstance(first_run_in_this_loo, RegressionRun):
                loo_run_type = "regression"
                this_loo_results = pd.DataFrame(columns=self.loo_full_regression_leaderboard_cols)
            else:
                raise TypeError("runs_in_this_loo[0] must be a ClassificationRun or RegressionRun object.")
            loo_id_folder_path = os.path.join(self.execution_id_folder_path, '{}_{}'.format("loo", this_loo_id))

            for run_object in runs_in_this_loo:
                run_id_folder_path = os.path.join(loo_id_folder_path, '{}_{}'.format("run", run_object.run_id))
                os.makedirs(run_id_folder_path, exist_ok=True)
                finished_run_result = self._output_single_run(run_object, run_id_folder_path)
                this_loo_results = this_loo_results.append(finished_run_result, ignore_index=True)

            loo_html_path = os.path.join(loo_id_folder_path, "loo_results.html")
            if loo_run_type == "classification":
                this_loo_results.sort_values('AUC Score', inplace=True, ascending=False)
                this_loo_results.reset_index(inplace=True, drop=True)
                loo_classification_results = loo_classification_results.append(this_loo_results)
            elif loo_run_type == "regression":
                this_loo_results.sort_values('R-Squared', inplace=True, ascending=False)
                this_loo_results.reset_index(inplace=True, drop=True)
                loo_regression_results = loo_regression_results.append(this_loo_results)

            this_loo_results.to_html(loo_html_path, index=False)

            if loo_run_type == "classification":
                this_loo_summary = pd.DataFrame(columns=self.loo_summarized_classification_leaderboard_cols)
                date_ran_summary = first_run_in_this_loo.date_ran
                time_ran_summary = first_run_in_this_loo.time_ran
                mean_auc_score = mean(this_loo_results['AUC Score'])
                std_auc_score = pstdev(this_loo_results['AUC Score'])
                mean_accuracy = mean(this_loo_results['Classification Accuracy'])
                std_accuracy = pstdev(this_loo_results['Classification Accuracy'])

                summary_values = {'Execution ID': self._execution_id, 'Leave-One-Out ID': this_loo_id,
                                  'Date': date_ran_summary, 'Time': time_ran_summary, 'Mean AUC Score': mean_auc_score,
                                  'Mean Classification Accuracy': mean_accuracy,
                                  'Model Description': first_run_in_this_loo.model_description,
                                  'Column Predicted': first_run_in_this_loo.col_to_predict,
                                  'Number Of Features Used': first_run_in_this_loo.num_features_used,
                                  'Data Description': first_run_in_this_loo.loo_dict["data_description"],
                                  'Grouping Description': first_run_in_this_loo.loo_dict["grouping_description"],
                                  'Normalized': first_run_in_this_loo.normalize,
                                  'Number of Features Normalized': first_run_in_this_loo.num_features_normalized,
                                  'Feature Extraction': first_run_in_this_loo.feature_extraction}
                this_loo_summary = this_loo_summary.append(summary_values, ignore_index=True)
                loo_classification_summaries = loo_classification_summaries.append(this_loo_summary, ignore_index=True)
            elif loo_run_type == "regression":
                this_loo_summary = pd.DataFrame(columns=self.loo_summarized_regression_leaderboard_cols)

                date_ran_summary = first_run_in_this_loo.date_ran
                time_ran_summary = first_run_in_this_loo.time_ran
                mean_rsquared = mean(this_loo_results['R-Squared'])
                std_rsquared = pstdev(this_loo_results['R-Squared'])
                mean_rmse = mean(this_loo_results['RMSE'])
                std_rmse = pstdev(this_loo_results['RMSE'])

                summary_values = {'Execution ID': self._execution_id, 'Leave-One-Out ID': this_loo_id,
                                  'Date': date_ran_summary, 'Time': time_ran_summary, 'Mean R-Squared': mean_rsquared,
                                  'Mean RMSE': mean_rmse, 'Model Description': first_run_in_this_loo.model_description,
                                  'Column Predicted': first_run_in_this_loo.col_to_predict,
                                  'Number Of Features Used': first_run_in_this_loo.num_features_used,
                                  'Data Description': first_run_in_this_loo.loo_dict["data_description"],
                                  'Grouping Description': first_run_in_this_loo.loo_dict["grouping_description"],
                                  'Normalized': first_run_in_this_loo.normalize,
                                  'Number of Features Normalized': first_run_in_this_loo.num_features_normalized,
                                  'Feature Extraction': first_run_in_this_loo.feature_extraction}
                this_loo_summary = this_loo_summary.append(summary_values, ignore_index=True)
                loo_regression_summaries = loo_regression_summaries.append(this_loo_summary, ignore_index=True)
            else:
                raise TypeError("run_object must be a ClassificationRun or RegressionRun object.")

        custom_classification_results.sort_values('AUC Score', inplace=True, ascending=False)
        custom_classification_results.reset_index(inplace=True, drop=True)
        html_path = os.path.join(self.execution_id_folder_path, "custom_classification_results.html")
        custom_classification_results.to_html(html_path, index=False, classes='custom_classification')

        custom_regression_results.sort_values('R-Squared', inplace=True, ascending=False)
        custom_regression_results.reset_index(inplace=True, drop=True)
        html_path = os.path.join(self.execution_id_folder_path, "custom_regression_results.html")
        custom_regression_results.to_html(html_path, index=False, classes='custom_regression')

        loo_classification_results.sort_values('AUC Score', inplace=True, ascending=False)
        loo_classification_results.reset_index(inplace=True, drop=True)
        html_path = os.path.join(self.execution_id_folder_path, "execution_classification_results.html")
        loo_classification_results.to_html(html_path, index=False, classes='loo_classification')

        loo_classification_summaries.sort_values('Mean AUC Score', inplace=True, ascending=False)
        loo_classification_summaries.reset_index(inplace=True, drop=True)
        html_path = os.path.join(self.execution_id_folder_path, "execution_classification_summaries.html")
        loo_classification_summaries.to_html(html_path, index=False, classes='loo_classification')

        loo_regression_results.sort_values('R-Squared', inplace=True, ascending=False)
        loo_regression_results.reset_index(inplace=True, drop=True)
        html_path = os.path.join(self.execution_id_folder_path, "execution_regression_results.html")
        loo_regression_results.to_html(html_path, index=False, classes='loo_regression')

        loo_regression_summaries.sort_values('Mean R-Squared', inplace=True, ascending=False)
        loo_regression_summaries.reset_index(inplace=True, drop=True)
        html_path = os.path.join(self.execution_id_folder_path, "execution_regression_summaries.html")
        loo_regression_summaries.to_html(html_path, index=False, classes='loo_regression')

        if len(custom_classification_results) > 0:
            print("\nCustom Classification Results:")
            print(custom_classification_results)
        if len(custom_regression_results) > 0:
            print("\nCustom Regression Results:")
            print(custom_regression_results)
        if len(loo_classification_summaries) > 0:
            print("\nLOO Classification Summaries:")
            print(loo_classification_summaries)
            print()
            print("\nLOO Classification Results:")
            print(loo_classification_results)
        if len(loo_regression_summaries) > 0:
            print("\nLOO Regression Summaries:")
            print(loo_regression_summaries)
            print("\nLOO Regression Results:")
            print(loo_regression_results)

        # Check if leaderboards exist, and create them if they don't
        # Pandas append docs: "Columns not in this frame are added as new columns" --> don't worry about adding new leaderboard cols
        cc_leaderboard_name = 'custom_classification_leaderboard'
        html_path = os.path.join(self.results_folder_path, "{}.html".format(cc_leaderboard_name))
        try:
            cc_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            cc_leaderboard = pd.DataFrame(columns=self.custom_classification_leaderboard_cols)
        cc_leaderboard = cc_leaderboard.append(custom_classification_results)
        cc_leaderboard.to_html(html_path, index=False, classes='comparable_classification')

        cr_leaderboard_name = 'custom_regression_leaderboard'
        html_path = os.path.join(self.results_folder_path, "{}.html".format(cr_leaderboard_name))
        try:
            cr_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            cr_leaderboard = pd.DataFrame(columns=self.custom_regression_leaderboard_cols)
        cr_leaderboard = cr_leaderboard.append(custom_regression_results)
        cr_leaderboard.to_html(html_path, index=False, classes='comparable_regression')

        lc_leaderboard_name = 'loo_summarized_classification_leaderboard'
        html_path = os.path.join(self.results_folder_path, "{}.html".format(lc_leaderboard_name))
        try:
            lc_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            lc_leaderboard = pd.DataFrame(columns=self.loo_summarized_classification_leaderboard_cols)
        lc_leaderboard = lc_leaderboard.append(loo_classification_summaries)
        lc_leaderboard.to_html(html_path, index=False, classes='loo_classification')

        lr_leaderboard_name = 'loo_summarized_regression_leaderboard'
        html_path = os.path.join(self.results_folder_path, "{}.html".format(lr_leaderboard_name))
        try:
            lr_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            lr_leaderboard = pd.DataFrame(columns=self.loo_summarized_regression_leaderboard_cols)
        lr_leaderboard = lr_leaderboard.append(loo_regression_summaries)
        lr_leaderboard.to_html(html_path, index=False, classes='loo_regression')

        lc_leaderboard_name = 'loo_full_classification_leaderboard'
        html_path = os.path.join(self.results_folder_path, "{}.html".format(lc_leaderboard_name))
        try:
            lc_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            lc_leaderboard = pd.DataFrame(columns=self.loo_full_classification_leaderboard_cols)
        lc_leaderboard = lc_leaderboard.append(loo_classification_results)
        lc_leaderboard.to_html(html_path, index=False, classes='loo_classification')

        lr_leaderboard_name = 'loo_full_regression_leaderboard'
        html_path = os.path.join(self.results_folder_path, "{}.html".format(lr_leaderboard_name))
        try:
            lr_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            lr_leaderboard = pd.DataFrame(columns=self.loo_full_regression_leaderboard_cols)
        lr_leaderboard = lr_leaderboard.append(loo_regression_results)
        lr_leaderboard.to_html(html_path, index=False, classes='loo_regression')

    def _output_single_run(self, run_object, output_path):
        if isinstance(run_object, ClassificationRun):
            row_values = {'Execution ID': self._execution_id, 'Run ID': run_object.run_id, 'Date': run_object.date_ran,
                          'Time': run_object.time_ran,
                          'AUC Score': run_object.auc_score, 'Classification Accuracy': run_object.percent_accuracy,
                          'Model Description': run_object.model_description, 'Column Predicted': run_object.col_to_predict,
                          'Number Of Features Used': run_object.num_features_used,
                          'Data and Split Description': run_object.data_and_split_description, 'Normalized': run_object.normalize,
                          'Number of Features Normalized': run_object.num_features_normalized,
                          'Feature Extraction': run_object.feature_extraction,
                          "Was Untested Data Predicted": run_object.was_untested_data_predicted}
            row_of_results = pd.DataFrame(columns=self.custom_classification_leaderboard_cols)
            row_of_results = row_of_results.append(row_values, ignore_index=True)
        elif isinstance(run_object, RegressionRun):
            row_values = {'Execution ID': self._execution_id, 'Run ID': run_object.run_id, 'Date': run_object.date_ran,
                          'Time': run_object.time_ran,
                          'R-Squared': run_object.r_squared, 'RMSE': run_object.rmse,
                          'Model Description': run_object.model_description, 'Column Predicted': run_object.col_to_predict,
                          'Number Of Features Used': run_object.num_features_used,
                          'Data and Split Description': run_object.data_and_split_description, 'Normalized': run_object.normalize,
                          'Number of Features Normalized': run_object.num_features_normalized,
                          'Feature Extraction': run_object.feature_extraction,
                          "Was Untested Data Predicted": run_object.was_untested_data_predicted}
            row_of_results = pd.DataFrame(columns=self.custom_regression_leaderboard_cols)
            row_of_results = row_of_results.append(row_values, ignore_index=True)
        else:
            raise ValueError()

        run_object.training_data.to_csv('{}/{}'.format(output_path, 'training_data.csv'), index=False)
        run_object.testing_data_predictions.to_csv('{}/{}'.format(output_path, 'testing_data.csv'), index=False)
        if run_object.was_untested_data_predicted is not False:
            prediction_data_to_save = run_object.untested_data_predictions.copy()
            prediction_data_to_save.to_csv('{}/{}'.format(output_path, 'predicted_data.csv'), index=False)
        if run_object.feature_extraction is not False:
            run_object.feature_importances.to_csv('{}/{}'.format(output_path, 'feature_importances.csv'), index=False)

        test_file_name = os.path.join(output_path, 'model_information.txt')
        with open(test_file_name, "w") as f:
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

        return row_of_results
