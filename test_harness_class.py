import os
import json
import time
import itertools
import pandas as pd
from test_harness.unique_id import get_id
from six import string_types
from datetime import datetime
from statistics import mean, pstdev
from test_harness.run_classes import ClassificationRun, RegressionRun
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


# TODO: think about removing the execution level


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
        self.runs_folder_path = os.path.join(self.results_folder_path, 'runs')
        if not os.path.exists(self.results_folder_path):
            os.makedirs(self.results_folder_path)
        if not os.path.exists(self.runs_folder_path):
            os.makedirs(self.runs_folder_path)

        # 'Normalized' should describe normalization method used (or False if no normalization)
        self.custom_classification_leaderboard_cols = \
            ['Run ID', 'Date', 'Time', 'AUC Score', 'Classification Accuracy', 'Model Description', 'Column Predicted',
             'Number Of Features Used', 'Data and Split Description', 'Normalized', 'Number of Features Normalized',
             'Feature Extraction', "Was Untested Data Predicted"]
        self.custom_regression_leaderboard_cols = \
            ['Run ID', 'Date', 'Time', 'R-Squared', 'RMSE', 'Model Description', 'Column Predicted',
             'Number Of Features Used', 'Data and Split Description', 'Normalized', 'Number of Features Normalized',
             'Feature Extraction', "Was Untested Data Predicted"]
        self.loo_full_classification_leaderboard_cols = \
            ['Leave-One-Out ID', 'Run ID', 'Date', 'Time', 'AUC Score', 'Classification Accuracy', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Test Group', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.loo_full_regression_leaderboard_cols = \
            ['Leave-One-Out ID', 'Run ID', 'Date', 'Time', 'R-Squared', 'RMSE', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Test Group', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.loo_summarized_classification_leaderboard_cols = \
            ['Leave-One-Out ID', 'Date', 'Time', 'Mean AUC Score', 'Mean Classification Accuracy', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Data Description', 'Grouping Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.loo_summarized_regression_leaderboard_cols = \
            ['Leave-One-Out ID', 'Date', 'Time', 'Mean R-Squared', 'Mean RMSE', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Data Description', 'Grouping Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction']
        self.leaderboard_names_dict = {"custom_classification_leaderboard": self.custom_classification_leaderboard_cols,
                                  "custom_regression_leaderboard": self.custom_regression_leaderboard_cols,
                                  "loo_summarized_classification_leaderboard": self.loo_summarized_classification_leaderboard_cols,
                                  "loo_summarized_regression_leaderboard": self.loo_summarized_regression_leaderboard_cols,
                                  "loo_detailed_classification_leaderboard": self.loo_full_classification_leaderboard_cols,
                                  "loo_detailed_regression_leaderboard": self.loo_full_regression_leaderboard_cols}
        self.metric_to_sort_classification_results_by = "AUC Score"
        self.metric_to_sort_regression_results_by = "R-Squared"
        self.valid_feature_extraction_methods = ['eli5_permutation', 'rfpimp_permutation']

    # TODO: add more normalization options: http://benalexkeen.com/feature-scaling-with-scikit-learn/
    # TODO: make feature_extraction options something like: "BBA", "permutation", and "custom", where custom means that
    # TODO: it's not a black box feature tool, but rather a specific one defined inside of the TestHarnessModel object
    def run_custom(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                   data_and_split_description, cols_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                   feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None):
        """
        Instantiates and runs a model on a custom train/test split
        If you pass in a list of columns to predict, a separate run will occur for each string in the list
        :param function_that_returns_TH_model:
        :param dict_of_function_parameters:
        :param training_data:
        :param testing_data:
        :param data_and_split_description:
        :param cols_to_predict:
        :param feature_cols_to_use:
        :param normalize:
        :param feature_cols_to_normalize:
        :param feature_extraction:
        :param predict_untested_data:
        :param sparse_cols_to_use:
        :return:
        """
        cols_to_predict = make_list_if_not_list(cols_to_predict)
        assert is_list_of_strings(cols_to_predict), "cols_to_predict must be a string or a list of strings"

        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        if feature_cols_to_normalize:
            feature_cols_to_normalize = make_list_if_not_list(feature_cols_to_normalize)
        if sparse_cols_to_use:
            sparse_cols_to_use = make_list_if_not_list(sparse_cols_to_use)

        for col in cols_to_predict:
            self._execute_run(function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                              data_and_split_description, col, feature_cols_to_use, normalize, feature_cols_to_normalize,
                              feature_extraction, predict_untested_data, sparse_cols_to_use)

    def run_leave_one_out(self, function_that_returns_TH_model, dict_of_function_parameters, data, data_description, grouping,
                          grouping_description, cols_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                          feature_extraction=False):
        """
        Splits the data into appropriate train/test splits according to the grouping dataframe, and then runs a separate instantiation of
        the passed-in model on each split.
        :param function_that_returns_TH_model:
        :param dict_of_function_parameters:
        :param data:
        :param data_description:
        :param grouping:
        :param grouping_description:
        :param cols_to_predict:
        :param feature_cols_to_use:
        :param normalize:
        :param feature_cols_to_normalize:
        :param feature_extraction:
        :return:
        """
        date_loo_ran = datetime.now().strftime("%Y-%m-%d")
        time_loo_ran = datetime.now().strftime("%H:%M:%S")

        cols_to_predict = make_list_if_not_list(cols_to_predict)
        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        if feature_cols_to_normalize:
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
            loo_folder_path = os.path.join(self.runs_folder_path, '{}_{}'.format("loo", loo_id))
            os.makedirs(loo_folder_path, exist_ok=False)
            data.to_csv(os.path.join(loo_folder_path, "data.csv"), index=False)
            grouping.to_csv(os.path.join(loo_folder_path, "grouping.csv"), index=False)

            dummy_th_model = function_that_returns_TH_model(**dict_of_function_parameters)
            if isinstance(dummy_th_model, ClassificationModel):
                task_type = "Classification"
            elif isinstance(dummy_th_model, RegressionModel):
                task_type = "Regression"
            else:
                raise ValueError("function_that_returns_TH_model must return a ClassificationModel or a RegressionModel.")

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

                loo_dict = {"loo_id": loo_id, "task_type": task_type, "data_description": data_description,
                            "grouping_description": grouping_description, "group_info": group_info}

                self._execute_run(function_that_returns_TH_model, dict_of_function_parameters, train_split, test_split,
                                  data_and_split_description, col, feature_cols_to_use, normalize, feature_cols_to_normalize,
                                  feature_extraction, False, None, loo_dict)

            # TODO: do summary results here, and update summary leaderboard
            if task_type == "Classification":
                detailed_leaderboard_name = "loo_detailed_classification_leaderboard"
                detailed_leaderboard_path = os.path.join(self.results_folder_path, "{}.html".format(detailed_leaderboard_name))
                detailed_leaderboard = pd.read_html(detailed_leaderboard_path)[0]
                this_loo_results = detailed_leaderboard.loc[detailed_leaderboard["Leave-One-Out ID"] == loo_id]

                mean_auc_score = mean(this_loo_results['AUC Score'])
                std_auc_score = pstdev(this_loo_results['AUC Score'])
                mean_accuracy = mean(this_loo_results['Classification Accuracy'])
                std_accuracy = pstdev(this_loo_results['Classification Accuracy'])

                summary_values = {'Leave-One-Out ID': loo_id,
                                  'Date': date_loo_ran, 'Time': time_loo_ran, 'Mean AUC Score': mean_auc_score,
                                  'Mean Classification Accuracy': mean_accuracy,
                                  'Model Description': dummy_th_model.model_description,
                                  'Column Predicted': col,
                                  'Number Of Features Used': len(feature_cols_to_use),
                                  'Data Description': data_description,
                                  'Grouping Description': grouping_description,
                                  'Normalized': normalize,
                                  'Number of Features Normalized': len(feature_cols_to_normalize),
                                  'Feature Extraction': feature_extraction}

                # Update summary leaderboard
                summary_leaderboard_name = "loo_summarized_classification_leaderboard"
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

            elif task_type == "Regression":
                detailed_leaderboard_name = "loo_detailed_regression_leaderboard"
                detailed_leaderboard_path = os.path.join(self.results_folder_path, "{}.html".format(detailed_leaderboard_name))
                detailed_leaderboard = pd.read_html(detailed_leaderboard_path)[0]
                this_loo_results = detailed_leaderboard.loc[detailed_leaderboard["Leave-One-Out ID"] == loo_id]

                mean_rsquared = mean(this_loo_results['R-Squared'])
                std_rsquared = pstdev(this_loo_results['R-Squared'])
                mean_rmse = mean(this_loo_results['RMSE'])
                std_rmse = pstdev(this_loo_results['RMSE'])

                summary_values = {'Leave-One-Out ID': loo_id,
                                  'Date': date_loo_ran, 'Time': time_loo_ran, 'Mean R-Squared': mean_rsquared,
                                  'Mean RMSE': mean_rmse, 'Model Description': dummy_th_model.model_description,
                                  'Column Predicted': col,
                                  'Number Of Features Used': len(feature_cols_to_use),
                                  'Data Description': data_description,
                                  'Grouping Description': grouping_description,
                                  'Normalized': normalize,
                                  'Number of Features Normalized': len(feature_cols_to_normalize),
                                  'Feature Extraction': feature_extraction}

                # Update summary leaderboard
                summary_leaderboard_name = "loo_summarized_regression_leaderboard"
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
                summary_leaderboard.sort_values(sort_metric, inplace=True, ascending=False)
                summary_leaderboard.reset_index(inplace=True, drop=True)

                # overwrite old leaderboard with updated leaderboard
                summary_leaderboard.to_html(html_path, index=False, classes=summary_leaderboard_name)
            else:
                raise TypeError("run_object must be a ClassificationRun or RegressionRun object.")


    # TODO: replace loo_dict with type_dict --> first entry is run type --> this will allow for more types in the future
    def _execute_run(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                     data_and_split_description, col_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                     feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None, loo_dict=False):
        """
        1. Instantiates the TestHarnessModel object
        2. Creates a ClassificationRun or RegressionRun object and calls their train_and_test_model and calculate_metrics methods
        3. Calls _output_results(Run Object)
        4. Calls _delete_run_object() to free up RAM

        :param function_that_returns_TH_model:
        :param dict_of_function_parameters:
        :param training_data:
        :param testing_data:
        :param data_and_split_description:
        :param col_to_predict:
        :param feature_cols_to_use:
        :param normalize:
        :param feature_cols_to_normalize:
        :param feature_extraction:
        :param predict_untested_data:
        :param sparse_cols_to_use:
        :param loo_dict:
        :return:
        """

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
            run_object = ClassificationRun(test_harness_model, train_df, test_df, data_and_split_description, col_to_predict,
                                           feature_cols_to_use, normalize, feature_cols_to_normalize, feature_extraction,
                                           predict_untested_data, loo_dict)
        elif isinstance(test_harness_model, RegressionModel):
            run_object = RegressionRun(test_harness_model, train_df, test_df, data_and_split_description, col_to_predict,
                                       feature_cols_to_use, normalize, feature_cols_to_normalize, feature_extraction, predict_untested_data,
                                       loo_dict)
        else:
            raise TypeError("test_harness_model must be a ClassificationModel or a RegressionModel.")

        # call run object methods
        start = time.time()
        print('Starting run at time {}'.format(datetime.now().strftime("%H:%M:%S")))
        run_object.train_and_test_model()
        run_object.calculate_metrics()
        if run_object.feature_extraction is not False:
            run_object.feature_extraction_method(method=run_object.feature_extraction)

        # output results of run object by updating the appropriate leaderboard(s) and writing files to disk


        # Pandas append docs: "Columns not in this frame are added as new columns" --> don't worry about adding new leaderboard cols

        self._update_leaderboard(run_object)

        if run_object.loo_dict is False:
            run_id_folder_path = os.path.join(self.runs_folder_path, '{}_{}'.format("run", run_object.run_id))
            os.makedirs(run_id_folder_path)
            self._output_run_files(run_object, run_id_folder_path, output_data_csvs=True)
        else:
            loo_id = run_object.loo_dict['loo_id']
            loo_path = os.path.join(self.runs_folder_path, '{}_{}'.format("loo", loo_id))
            os.makedirs(loo_path, exist_ok=True)
            run_id_folder_path = os.path.join(loo_path, '{}_{}'.format("run", run_object.run_id))
            os.makedirs(run_id_folder_path)
            self._output_run_files(run_object, run_id_folder_path, output_data_csvs=True)

        end = time.time()
        print('Run finished at {}'.format(datetime.now().strftime("%H:%M:%S")), 'Total run time = {0:.2f} seconds'.format(end - start))
        print()

    def _update_leaderboard(self, run_object):
        # find appropriate leaderboard to update based on run_object characteristics
        if run_object.loo_dict is False:
            # TODO: Hamed look up data structures tree unit and see if it's a good way to connect leaderboards to run objs
            if isinstance(run_object, ClassificationRun):
                leaderboard_name = "custom_classification_leaderboard"
            elif isinstance(run_object, RegressionRun):
                leaderboard_name = "custom_regression_leaderboard"
            else:
                raise TypeError("run_object must be a ClassificationRun or a RegressionRun")
        else:
            if isinstance(run_object, ClassificationRun):
                leaderboard_name = "loo_detailed_classification_leaderboard"
            elif isinstance(run_object, RegressionRun):
                leaderboard_name = "loo_detailed_regression_leaderboard"
            else:
                raise TypeError("run_object must be a ClassificationRun or a RegressionRun")
        assert  leaderboard_name in self.leaderboard_names_dict.keys(), "passed-in leaderboard_name is not valid."
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
            row_of_results["Leave-One-Out ID"] = run_object.loo_dict["loo_id"]
            row_of_results["Test Group"] = run_object.loo_dict["group_info"]
        print()
        print(row_of_results)
        print()

        # update leaderboard with new entry (row_of_results) and sort it based on run type
        leaderboard = leaderboard.append(row_of_results, ignore_index=True, sort=False)
        if isinstance(run_object, ClassificationRun):
            leaderboard.sort_values(self.metric_to_sort_classification_results_by, inplace=True, ascending=False)
        elif isinstance(run_object, RegressionRun):
            leaderboard.sort_values(self.metric_to_sort_regression_results_by, inplace=True, ascending=False)
        else:
            raise TypeError("run_object must be a ClassificationRun or RegressionRun object.")
        leaderboard.reset_index(inplace=True, drop=True)

        # overwrite old leaderboard with updated leaderboard
        leaderboard.to_html(html_path, index=False, classes=leaderboard_name)

    def _create_row_entry(self, run_object):
        if isinstance(run_object, ClassificationRun):
            row_values = {'Run ID': run_object.run_id, 'Date': run_object.date_ran,
                          'Time': run_object.time_ran,
                          'AUC Score': run_object.auc_score, 'Classification Accuracy': run_object.percent_accuracy,
                          'Model Description': run_object.model_description, 'Column Predicted': run_object.col_to_predict,
                          'Number Of Features Used': run_object.num_features_used,
                          'Data and Split Description': run_object.data_and_split_description, 'Normalized': run_object.normalize,
                          'Number of Features Normalized': run_object.num_features_normalized,
                          'Feature Extraction': run_object.feature_extraction,
                          "Was Untested Data Predicted": run_object.was_untested_data_predicted}
            row_of_results = pd.DataFrame(columns=self.custom_classification_leaderboard_cols)
            row_of_results = row_of_results.append(row_values, ignore_index=True, sort=False)
        elif isinstance(run_object, RegressionRun):
            row_values = {'Run ID': run_object.run_id, 'Date': run_object.date_ran,
                          'Time': run_object.time_ran,
                          'R-Squared': run_object.r_squared, 'RMSE': run_object.rmse,
                          'Model Description': run_object.model_description, 'Column Predicted': run_object.col_to_predict,
                          'Number Of Features Used': run_object.num_features_used,
                          'Data and Split Description': run_object.data_and_split_description, 'Normalized': run_object.normalize,
                          'Number of Features Normalized': run_object.num_features_normalized,
                          'Feature Extraction': run_object.feature_extraction,
                          "Was Untested Data Predicted": run_object.was_untested_data_predicted}
            row_of_results = pd.DataFrame(columns=self.custom_regression_leaderboard_cols)
            row_of_results = row_of_results.append(row_values, ignore_index=True, sort=False)
        else:
            raise ValueError()
        return row_of_results


    def _output_run_files(self, run_object, output_path, output_data_csvs=True):
        if output_data_csvs:
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


    def print_leaderboards(self):
        pass


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



