import os
import json
import time
import itertools
import pandas as pd
from unique_id import get_id
from six import string_types
from datetime import datetime
from sklearn import preprocessing
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


def is_list_of_TH_models(obj):
    if obj and isinstance(obj, list):
        return all(isinstance(elem, TestHarnessModel) for elem in obj)
    else:
        return False


class TestHarness:
    def __init__(self, output_path=os.path.dirname(os.path.realpath(__file__))):
        # Note: loo stands for leave-one-out
        self.output_path = output_path
        self._custom_runs_to_execute = []
        self._loo_runs_to_execute = []
        self._execution_id = None
        self._finished_custom_runs = []
        self._finished_loo_runs = []
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
             'Column Predicted', 'Number Of Features Used', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction', "Was Untested Data Predicted"]
        self.loo_full_regression_leaderboard_cols = \
            ['Execution ID', 'Leave-One-Out ID', 'Run ID', 'Date', 'Time', 'R-Squared', 'RMSE', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction', "Was Untested Data Predicted"]
        self.loo_summarized_classification_leaderboard_cols = \
            ['Execution ID', 'Leave-One-Out ID', 'Date', 'Time', 'Mean AUC Score', 'Mean Classification Accuracy', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction', "Was Untested Data Predicted"]
        self.loo_summarized_regression_leaderboard_cols = \
            ['Execution ID', 'Leave-One-Out ID', 'Date', 'Time', 'Mean R-Squared', 'Mean RMSE', 'Model Description',
             'Column Predicted', 'Number Of Features Used', 'Data and Split Description', 'Normalized',
             'Number of Features Normalized', 'Feature Extraction', "Was Untested Data Predicted"]

    # TODO: add more normalization options: http://benalexkeen.com/feature-scaling-with-scikit-learn/
    # TODO: make feature_extraction options something like: "BBA", "permutation", and "custom", where custom means that
    # TODO: it's not a black box feature tool, but rather a specific one defined inside of the TestHarnessModel object
    def add_custom_runs(self, test_harness_models, training_data, testing_data, data_and_split_description,
                        cols_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                        feature_extraction=False, predict_untested_data=False):
        # Adds custom run(s) to the TestHarness object
        # If you pass a list of models and/or list of columns to predict, a custom run will be added for every
        # combination of models and columns to predict that you provided.
        # Custom runs require providing:
        #       - a TestHarnessModel or list of TestHarnessModels
        #       - a training dataframe and a testing dataframe
        #       - a column to predict or list of columns to predict
        #       - other arguments
        test_harness_models = make_list_if_not_list(test_harness_models)
        cols_to_predict = make_list_if_not_list(cols_to_predict)
        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        feature_cols_to_normalize = make_list_if_not_list(feature_cols_to_normalize)

        # Single strings are included in the assert error messages because the make_list_if_not_list function was used
        assert is_list_of_TH_models(
            test_harness_models), "test_harness_models must be a TestHarnessModel object or a list of TestHarnessModel objects"
        assert isinstance(training_data, pd.DataFrame), "training_data must be a Pandas Dataframe"
        assert isinstance(testing_data, pd.DataFrame), "testing_data must be a Pandas Dataframe"
        assert isinstance(data_and_split_description, str), "data_and_split_description must be a string"
        assert is_list_of_strings(cols_to_predict), "cols_to_predict must be a string or a list of strings"
        assert is_list_of_strings(feature_cols_to_use), "feature_cols_to_use must be a string or a list of strings"
        assert isinstance(normalize, bool), "normalize must be True or False"
        assert (feature_cols_to_normalize is None) or is_list_of_strings(feature_cols_to_normalize), \
            "feature_cols_to_normalize must be None, a string, or a list of strings"
        assert isinstance(feature_extraction, bool), "feature_extraction must be True or False"
        assert (predict_untested_data == False) or (isinstance(predict_untested_data, pd.DataFrame)), \
            "predict_untested_data must be False or a Pandas Dataframe"

        for combo in itertools.product(test_harness_models, cols_to_predict):
            test_harness_model = combo[0]
            col_to_predict = combo[1]
            custom_run_dict = {"test_harness_model": test_harness_model, "training_data": training_data,
                               "testing_data": testing_data, "data_and_split_description": data_and_split_description,
                               "col_to_predict": col_to_predict, "feature_cols_to_use": feature_cols_to_use,
                               "normalize": normalize, "feature_cols_to_normalize": feature_cols_to_normalize,
                               "feature_extraction": feature_extraction, "predict_untested_data": predict_untested_data}
            self._custom_runs_to_execute.append(custom_run_dict)

    def add_leave_one_out_runs(self, test_harness_models, data, data_description, grouping, grouping_description,
                               cols_to_predict, feature_cols_to_use, normalize=False, feature_cols_to_normalize=None,
                               feature_extraction=False, predict_untested_data=False):
        # Adds leave-one-out run(s) to the TestHarness object
        # Leave-one-out runs require providing:
        #       - a TestHarnessModel or list of TestHarnessModels
        #       - a dataset dataframe
        #       - a grouping dataframe or a list of column names to group by
        #       - a column to predict or list of columns to predict
        #       - other arguments
        test_harness_models = make_list_if_not_list(test_harness_models)
        cols_to_predict = make_list_if_not_list(cols_to_predict)
        feature_cols_to_use = make_list_if_not_list(feature_cols_to_use)
        feature_cols_to_normalize = make_list_if_not_list(feature_cols_to_normalize)

        assert is_list_of_TH_models(
            test_harness_models), "test_harness_models must be a TestHarnessModel object or a list of TestHarnessModel objects"
        assert isinstance(data, pd.DataFrame), "data must be a Pandas Dataframe"
        assert isinstance(data_description, str), "data_description must be a string"
        assert isinstance(grouping, pd.DataFrame) or is_list_of_strings(
            grouping), "grouping must be a Pandas Dataframe or a list of column names"
        assert isinstance(grouping_description, str), "grouping_description must be a string"
        assert is_list_of_strings(cols_to_predict), "cols_to_predict must be a string or a list of strings"
        assert is_list_of_strings(feature_cols_to_use), "feature_cols_to_use must be a string or a list of strings"
        assert isinstance(normalize, bool), "normalize must be True or False"
        assert (feature_cols_to_normalize is None) or is_list_of_strings(feature_cols_to_normalize), \
            "feature_cols_to_normalize must be None, a string, or a list of strings"
        assert isinstance(feature_extraction, bool), "feature_extraction must be True or False"
        assert (predict_untested_data == False) or (isinstance(predict_untested_data, pd.DataFrame)), \
            "predict_untested_data must be False or a Pandas Dataframe"

        for combo in itertools.product(test_harness_models, cols_to_predict):
            test_harness_model = combo[0]
            col_to_predict = combo[1]
            loo_run_dict = {"test_harness_model": test_harness_model, "data": data,
                            "data_description": data_description, "grouping": grouping,
                            "grouping_description": grouping_description, "col_to_predict": col_to_predict,
                            "feature_cols_to_use": feature_cols_to_use, "normalize": normalize,
                            "feature_cols_to_normalize": feature_cols_to_normalize,
                            "feature_extraction": feature_extraction, "predict_untested_data": predict_untested_data}
            self._loo_runs_to_execute.append(loo_run_dict)

    def view_added_runs(self):
        print("Custom Runs Added:")
        print(self._custom_runs_to_execute)
        print()
        print("Leave-One-Out Runs Added:")
        print(self._loo_runs_to_execute)

    # Executes runs that have been added to self._custom_runs_to_execute and self._loo_runs_to_execute
    # by using _execute_custom_run and _execute_leave_one_out_run
    def execute_runs(self):
        self._execution_id = get_id()
        print("The ID for this Execution of runs is: {}".format(self._execution_id))
        print()

        # TODO: figure out how to prevent simultaneous leaderboard updates from overwriting each other
        number_of_custom_runs = len(self._custom_runs_to_execute)
        if number_of_custom_runs > 0:
            print("Executing {} custom runs".format(number_of_custom_runs))
            print()
            for counter, custom_run in enumerate(self._custom_runs_to_execute, start=1):
                start = time.time()
                print('Starting custom run {}/{} at time {}'.format(counter, number_of_custom_runs,
                                                                    datetime.now().strftime("%H:%M:%S")))
                self._execute_custom_run(**custom_run)
                end = time.time()
                print('Custom run finished at {}'.format(datetime.now().strftime("%H:%M:%S")))
                print('Total run time = {0:.2f} seconds'.format(end - start))
                print()

        number_of_loo_runs = len(self._loo_runs_to_execute)
        if number_of_loo_runs > 0:
            print("Executing {} leave-one-out runs".format(number_of_loo_runs))
            print()
            for counter, loo_run in enumerate(self._loo_runs_to_execute, start=1):
                start = time.time()
                print('Starting leave-one-out run {}/{} at time {}'.format(counter, number_of_loo_runs,
                                                                           datetime.now().strftime("%H:%M:%S")))
                self._execute_leave_one_out_run(**loo_run)
                end = time.time()
                print('Leave-one-out run finished at {}'.format(datetime.now().strftime("%H:%M:%S")))
                print('Total run time = {0:.2f} seconds'.format(end - start))
                print()

        print("Outputting results from all executed runs...")
        self._output_results()

    def _normalize_dataframes(self, training_data, testing_data, normalize, feature_cols_to_normalize, predict_untested_data=False):
        if feature_cols_to_normalize is None:
            raise ValueError("if normalize is True, then feature_cols_to_normalize must be a list of column names")

        train_df = training_data.copy()
        test_df = testing_data.copy()

        print("Normalizing training and testing splits...")
        if (normalize is True) or (normalize == "StandardScaler"):
            scaler = preprocessing.StandardScaler().fit(train_df[feature_cols_to_normalize])
        elif normalize == "MinMax":
            raise ValueError("MinMax normalization hasn't been added yet")
        else:
            raise ValueError("normalize must have a value of True, 'StandardScaler', or 'MinMax'")

        normalized_train = train_df.copy()
        normalized_train[feature_cols_to_normalize] = scaler.transform(normalized_train[feature_cols_to_normalize])
        normalized_test = test_df.copy()
        normalized_test[feature_cols_to_normalize] = scaler.transform(normalized_test[feature_cols_to_normalize])
        # Normalizing untested dataset if applicable
        if predict_untested_data is not False:
            normalized_untested = predict_untested_data.copy()
            normalized_untested[feature_cols_to_normalize] = scaler.transform(normalized_untested[feature_cols_to_normalize])
        else:
            normalized_untested = False

        return normalized_train, normalized_test, normalized_untested

    # Executes custom runs
    def _execute_custom_run(self, test_harness_model, training_data, testing_data, data_and_split_description,
                            col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                            feature_extraction, predict_untested_data):
        if normalize is not False:
            train_df, test_df, untested_df = self._normalize_dataframes(training_data, testing_data, normalize, feature_cols_to_normalize,
                                                                        predict_untested_data)
        else:
            train_df, test_df = training_data.copy(), testing_data.copy()
            if predict_untested_data is not False:
                untested_df = predict_untested_data.copy()
            else:
                untested_df = False

        if isinstance(test_harness_model, ClassificationModel):
            classification_run = ClassificationRun(test_harness_model, train_df, test_df, data_and_split_description, col_to_predict,
                                                   feature_cols_to_use, normalize, feature_cols_to_normalize, feature_extraction,
                                                   untested_df)
            classification_run.train_and_test()
            classification_run.calculate_metrics()
            self._finished_custom_runs.append(classification_run)
        elif isinstance(test_harness_model, RegressionModel):
            regression_run = RegressionRun(test_harness_model, train_df, test_df, data_and_split_description, col_to_predict,
                                           feature_cols_to_use, normalize, feature_cols_to_normalize, feature_extraction,
                                           untested_df)
            regression_run.train_and_test()
            regression_run.calculate_metrics()
            self._finished_custom_runs.append(regression_run)
        else:
            raise TypeError("test_harness_model must be a ClassificationModel or RegressionModel object.")

    # Executes leave-one-out runs
    def _execute_leave_one_out_run(self, function_that_returns_model_runner, all_data_df, grouping_df, col_to_predict,
                                   data_set_description, train_test_split_description="leave-one-group-out",
                                   normalize=False, feature_cols_to_normalize=None, get_pimportances=False,
                                   performance_output_path=None, features_output_path=None):
        if not callable(function_that_returns_model_runner):
            raise ValueError('function_that_returns_model_runner must be a function.')

        grouping_df = grouping_df.rename(columns={'name': 'topology'})

        all_data = all_data_df.copy()

        relevant_groupings = grouping_df.copy()
        relevant_groupings = relevant_groupings.loc[(relevant_groupings['dataset'].isin(all_data['dataset'])) &
                                                    (relevant_groupings['topology'].isin(all_data['topology']))]
        print(relevant_groupings)
        print()

        splits_results = pd.DataFrame()
        splits_features = None
        for group in list(set(relevant_groupings['group_index'])):
            train_split = all_data.copy()
            test_split = all_data.copy()
            print("Creating test split based on group {}:".format(group))
            group_df = relevant_groupings.loc[relevant_groupings['group_index'] == group]
            print(group_df.to_string(index=False))
            train_split = train_split.loc[~((train_split['dataset'].isin(group_df['dataset'])) &
                                            (train_split['topology'].isin(group_df['topology'])))]
            test_split = test_split.loc[(test_split['dataset'].isin(group_df['dataset'])) &
                                        (test_split['topology'].isin(group_df['topology']))]

            if normalize is True:
                if feature_cols_to_normalize is None:
                    raise ValueError(
                        "if normalize is True, then feature_cols_to_normalize must be a list of column names")
                print("Normalizing training and testing splits...")
                scaler = preprocessing.StandardScaler().fit(train_split[feature_cols_to_normalize])
                normalized_train = train_split.copy()
                normalized_train[feature_cols_to_normalize] = scaler.transform(
                    normalized_train[feature_cols_to_normalize])
                normalized_test = test_split.copy()
                normalized_test[feature_cols_to_normalize] = scaler.transform(
                    normalized_test[feature_cols_to_normalize])
                train_split, test_split = normalized_train.copy(), normalized_test.copy()

            print("Number of samples in train split:", train_split.shape)
            print("Number of samples in test split:", test_split.shape)

            model_runner = function_that_returns_model_runner(training_data=train_split,
                                                              testing_data=test_split,
                                                              col_to_predict=col_to_predict,
                                                              data_set_description=data_set_description,
                                                              train_test_split_description=train_test_split_description)

            this_run_results = model_runner.run_model()
            this_run_results['test_split'] = str(list(set(group_df['dataset'])) + list(set(group_df['topology'])))
            this_run_results['num_proteins_in_test_set'] = len(test_split)
            cols = list(this_run_results)
            cols.insert(1, cols.pop(cols.index('num_proteins_in_test_set')))
            cols.insert(1, cols.pop(cols.index('test_split')))
            this_run_results = this_run_results[cols]
            print(this_run_results)
            splits_results = pd.concat([splits_results, this_run_results])

            if get_pimportances is True:
                this_run_perms = model_runner.permutation_importances
                this_run_perms.rename(
                    columns={'Importance': str(list(set(group_df['dataset'])) + list(set(group_df['topology'])))},
                    inplace=True)
                if isinstance(this_run_perms, pd.DataFrame):
                    if splits_features is None:
                        splits_features = this_run_perms
                    else:
                        splits_features = pd.merge(splits_features, this_run_perms, on='Feature')
            print()

        # Sort by R Squared or AUC depending on regression/classification... update this
        splits_results = splits_results.sort_values('AUC Score', ascending=False)
        print(splits_results)
        print()
        print(splits_features)
        if performance_output_path is not None:
            print(performance_output_path)
            splits_results.to_csv(performance_output_path, index=False)
        if splits_features is not None and features_output_path is not None:
            splits_features.to_csv(features_output_path, index=False)

    # Outputs results (leaderboards, model outputs, etc) in a consistent way
    # 1. Create execution-specific leaderboard for this execution while saving outputs (including exec-specific leaderboard) to right folder
    # 2. Update main leaderboard to include results in this execution-specific leaderboard
    def _output_results(self):
        results_folder_path = os.path.join(self.output_path, 'results')
        execution_id_folder_path = os.path.join(results_folder_path, 'executions/{}'.format(self._execution_id))

        custom_classification_results = pd.DataFrame(columns=self.custom_classification_leaderboard_cols)
        custom_regression_results = pd.DataFrame(columns=self.custom_regression_leaderboard_cols)
        loo_full_classification_results = pd.DataFrame(columns=self.loo_full_classification_leaderboard_cols)
        loo_full_regression_results = pd.DataFrame(columns=self.loo_full_regression_leaderboard_cols)
        loo_summarized_classification_results = pd.DataFrame(columns=self.loo_summarized_classification_leaderboard_cols)
        loo_summarized_classification_results = pd.DataFrame(columns=self.loo_summarized_regression_leaderboard_cols)

        for fcr in self._finished_custom_runs:
            if isinstance(fcr, ClassificationRun):
                row_values = {'Execution ID': self._execution_id, 'Run ID': fcr.run_id, 'Date': fcr.date_ran, 'Time': fcr.time_ran,
                              'AUC Score': fcr.auc_score, 'Classification Accuracy': fcr.percent_accuracy,
                              'Model Description': fcr.model_description, 'Column Predicted': fcr.col_to_predict,
                              'Number Of Features Used': fcr.num_features_used,
                              'Data and Split Description': fcr.data_and_split_description, 'Normalized': fcr.normalize,
                              'Number of Features Normalized': fcr.num_features_normalized,
                              'Feature Extraction': fcr.feature_extraction,
                              "Was Untested Data Predicted": fcr.was_untested_data_predicted}
                custom_classification_results = custom_classification_results.append(row_values, ignore_index=True)
            elif isinstance(fcr, RegressionRun):
                row_values = {'Execution ID': self._execution_id, 'Run ID': fcr.run_id, 'Date': fcr.date_ran, 'Time': fcr.time_ran,
                              'R-Squared': fcr.r_squared, 'RMSE': fcr.rmse,
                              'Model Description': fcr.model_description, 'Column Predicted': fcr.col_to_predict,
                              'Number Of Features Used': fcr.num_features_used,
                              'Data and Split Description': fcr.data_and_split_description, 'Normalized': fcr.normalize,
                              'Number of Features Normalized': fcr.num_features_normalized,
                              'Feature Extraction': fcr.feature_extraction,
                              "Was Untested Data Predicted": fcr.was_untested_data_predicted}
                custom_regression_results = custom_regression_results.append(row_values, ignore_index=True)
            else:
                raise ValueError()

            training_data_to_save = fcr.training_data.copy()
            testing_data_to_save = fcr.testing_data_predictions.copy()
            if fcr.was_untested_data_predicted is not False:
                prediction_data_to_save = fcr.untested_data_predictions.copy()

            run_id_folder_path = os.path.join(execution_id_folder_path, '{}_{}'.format("run", fcr.run_id))
            os.makedirs(run_id_folder_path, exist_ok=True)
            training_data_to_save.to_csv('{}/{}'.format(run_id_folder_path, 'training_data.csv'), index=False)
            testing_data_to_save.to_csv('{}/{}'.format(run_id_folder_path, 'testing_data.csv'), index=False)
            if fcr.was_untested_data_predicted is not False:
                prediction_data_to_save.to_csv('{}/{}'.format(run_id_folder_path, 'predicted_data.csv'), index=False)
            test_file_name = os.path.join(run_id_folder_path, 'model_information.txt')
            with open(test_file_name, "w") as f:
                f.write("Feature columns used by model: \n")
                json.dump(fcr.feature_cols_to_use, f)
                f.write("\n\n\n")

                f.write("Model Instantiation Trace:\n")
                for i, t in enumerate(fcr.stack_trace):
                    f.write(" Level {}\n".format(i))
                    path, line, func = t[1:4]
                    f.write(' - Path: ' + path + '\n')
                    f.write(' - Line: ' + str(line) + ',  Function: ' + str(func) + '\n')
                    f.write("\n")

        custom_classification_results.sort_values('AUC Score', inplace=True, ascending=False)
        custom_classification_results.reset_index(inplace=True, drop=True)
        html_path = os.path.join(execution_id_folder_path, "custom_classification_results.html")
        custom_classification_results.to_html(html_path, index=False, classes='custom_classification')

        custom_regression_results.sort_values('R-Squared', inplace=True, ascending=False)
        custom_regression_results.reset_index(inplace=True, drop=True)
        html_path = os.path.join(execution_id_folder_path, "custom_regression_results.html")
        custom_regression_results.to_html(html_path, index=False, classes='custom_regression')

        if len(custom_classification_results) > 0:
            print()
            print(custom_classification_results)
        if len(custom_regression_results) > 0:
            print()
            print(custom_regression_results)

        # Check if leaderboards exist, and create them if they don't
        # Pandas append docs: "Columns not in this frame are added as new columns" --> don't worry about adding new leaderboard cols
        cc_leaderboard_name = 'custom_classification_leaderboard'
        html_path = os.path.join(results_folder_path, "{}.html".format(cc_leaderboard_name))
        try:
            cc_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            cc_leaderboard = pd.DataFrame(columns=self.custom_classification_leaderboard_cols)
        cc_leaderboard = cc_leaderboard.append(custom_classification_results)
        cc_leaderboard.to_html(html_path, index=False, classes='comparable_classification')

        cr_leaderboard_name = 'custom_regression_leaderboard'
        html_path = os.path.join(results_folder_path, "{}.html".format(cr_leaderboard_name))
        try:
            cr_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            cr_leaderboard = pd.DataFrame(columns=self.custom_regression_leaderboard_cols)
        cr_leaderboard = cr_leaderboard.append(custom_regression_results)
        cr_leaderboard.to_html(html_path, index=False, classes='comparable_regression')
