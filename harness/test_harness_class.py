import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from harness.unique_id import get_id
from six import string_types
from datetime import datetime
from statistics import mean
from sklearn.externals import joblib
from harness.run_classes import _BaseRun
from harness.test_harness_models_abstract_classes import ClassificationModel, RegressionModel
from harness.utils.names import Names

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
    def __init__(self, output_location=os.path.dirname(os.path.realpath(__file__)), output_csvs_of_leaderboards=False):
        # Note: loo stands for leave-one-out
        self.output_path = output_location
        self.output_csvs_of_leaderboards = output_csvs_of_leaderboards
        self.results_folder_path = os.path.join(self.output_path, 'test_harness_results')
        self.runs_folder_path = os.path.join(self.results_folder_path, 'runs')
        if not os.path.exists(self.results_folder_path):
            os.makedirs(self.results_folder_path, exist_ok=True)
        if not os.path.exists(self.runs_folder_path):
            os.makedirs(self.runs_folder_path, exist_ok=True)

        # add metrics here:
        self.classification_metrics = [Names.ACCURACY, Names.BALANCED_ACCURACY, Names.AUC_SCORE, Names.AVERAGE_PRECISION,
                                       Names.F1_SCORE, Names.PRECISION, Names.RECALL]
        self.mean_classification_metrics = ["Mean " + cm for cm in self.classification_metrics]
        self.regression_metrics = [Names.R_SQUARED, Names.RMSE]
        self.mean_regression_metrics = ["Mean " + rm for rm in self.regression_metrics]

        self.metric_to_sort_classification_results_by = Names.AVERAGE_PRECISION
        self.metric_to_sort_regression_results_by = Names.R_SQUARED

        custom_cols_1 = [Names.RUN_ID, Names.DATE, Names.TIME, Names.MODEL_NAME, Names.MODEL_AUTHOR]
        custom_cols_2 = [Names.SAMPLES_IN_TRAIN, Names.SAMPLES_IN_TEST, Names.MODEL_DESCRIPTION, Names.COLUMN_PREDICTED,
                         Names.NUM_FEATURES_USED, Names.DATA_AND_SPLIT_DESCRIPTION, Names.NORMALIZED, Names.NUM_FEATURES_NORMALIZED,
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

    # TODO: add more normalization options: http://benalexkeen.com/feature-scaling-with-scikit-learn/
    def run_custom(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                   data_and_split_description, cols_to_predict, feature_cols_to_use, index_cols=("dataset", "name"), normalize=False,
                   feature_cols_to_normalize=None, feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None):
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
        :param index_cols:
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
                              data_and_split_description, col, feature_cols_to_use, index_cols, normalize, feature_cols_to_normalize,
                              feature_extraction, predict_untested_data, sparse_cols_to_use)

    def run_leave_one_out(self, function_that_returns_TH_model, dict_of_function_parameters, data, data_description, grouping,
                          grouping_description, cols_to_predict, feature_cols_to_use, index_cols=("dataset", "name"), normalize=False,
                          feature_cols_to_normalize=None, feature_extraction=False):
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
        :param index_cols:
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
            num_features_normalized = len(feature_cols_to_normalize)
        else:
            num_features_normalized = 0

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
                data_and_split_description = "{}".format(data_description)
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
                                  data_and_split_description, col, feature_cols_to_use, index_cols, normalize, feature_cols_to_normalize,
                                  feature_extraction, False, None, loo_dict)

            # summary results are calculated here, and summary leaderboards are updated
            summary_values = {Names.LOO_ID: loo_id, Names.DATE: date_loo_ran, Names.TIME: time_loo_ran,
                              Names.MODEL_NAME: dummy_th_model.model_name, Names.MODEL_AUTHOR: dummy_th_model.model_author,
                              Names.MODEL_DESCRIPTION: dummy_th_model.model_description, Names.COLUMN_PREDICTED: col,
                              Names.NUM_FEATURES_USED: len(feature_cols_to_use), Names.DATA_DESCRIPTION: data_description,
                              Names.GROUPING_DESCRIPTION: grouping_description, Names.NORMALIZED: normalize,
                              Names.NUM_FEATURES_NORMALIZED: num_features_normalized, Names.FEATURE_EXTRACTION: feature_extraction}
            if task_type == "Classification":
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

            elif task_type == "Regression":
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
                print(summary_leaderboard)
                summary_leaderboard.sort_values(sort_metric, inplace=True, ascending=False)
                summary_leaderboard.reset_index(inplace=True, drop=True)

                # overwrite old leaderboard with updated leaderboard
                summary_leaderboard.to_html(html_path, index=False, classes=summary_leaderboard_name)
                if self.output_csvs_of_leaderboards is True:
                    csv_path = os.path.join(self.results_folder_path, "{}.csv".format(summary_leaderboard_name))
                    summary_leaderboard.to_csv(csv_path, index=False)

            else:
                raise TypeError("task_type must be 'Classification' or 'Regression'.")

    # TODO: replace loo_dict with type_dict --> first entry is run type --> this will allow for more types in the future
    def _execute_run(self, function_that_returns_TH_model, dict_of_function_parameters, training_data, testing_data,
                     data_and_split_description, col_to_predict, feature_cols_to_use, index_cols=("dataset", "name"), normalize=False,
                     feature_cols_to_normalize=None, feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None,
                     loo_dict=False):
        """
        1. Instantiates the TestHarnessModel object
        2. Creates a _BaseRun object and calls their train_and_test_model and calculate_metrics methods
        3. Calls _output_results(Run Object)

        :param function_that_returns_TH_model:
        :param dict_of_function_parameters:
        :param training_data:
        :param testing_data:
        :param data_and_split_description:
        :param col_to_predict:
        :param feature_cols_to_use:
        :param normalize:
        :param index_cols:
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
        assert (index_cols is None) or (isinstance(index_cols, list)) or (isinstance(index_cols, tuple)), \
            "index_cols must be None or a list (or tuple) of index column names in the passed-in training, testing, and prediction data."
        if isinstance(index_cols, tuple):
            index_cols = list(index_cols)
        if isinstance(index_cols, list):
            assert is_list_of_strings(index_cols), "if index_cols is a tuple or list, it must contain only strings."

        # check if index_cols exist in training, testing, and prediction dataframes:
        assert (set(index_cols).issubset(training_data.columns.tolist())), \
            "the strings in index_cols are not valid columns in training_data."
        assert (set(index_cols).issubset(testing_data.columns.tolist())), \
            "the strings in index_cols are not valid columns in testing_data."
        if isinstance(predict_untested_data, pd.DataFrame):
            assert (set(index_cols).issubset(predict_untested_data.columns.tolist())), \
                "the strings in index_cols are not valid columns in predict_untested_data."

        # TODO: add checks to ensure index_cols represent unique values in training, testing, and prediction dataframes

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

        # TODO sparse_cols for untested data
        # TODO move sparse col code to run_classes.py
        if sparse_cols_to_use is not None:
            train_df, feature_cols_to_use = self._make_sparse_cols(train_df, sparse_cols_to_use, feature_cols_to_use)
            test_df = self._make_sparse_cols(test_df, sparse_cols_to_use)

        test_harness_model = function_that_returns_TH_model(**dict_of_function_parameters)

        # This is the one and only time _BaseRun is invoked
        run_object = _BaseRun(test_harness_model, train_df, test_df, data_and_split_description, col_to_predict,
                              feature_cols_to_use, index_cols, normalize, feature_cols_to_normalize, feature_extraction,
                              pred_df, loo_dict)

        # tracking the run_ids of all the runs that were kicked off in this TestHarness instance
        # TODO: take into account complications when dealing with LOO runs. e.g. do we want to keep a list of LOO Ids as well (if yes, how).
        self.list_of_this_instance_run_ids.append(run_object.run_id)

        # call run object methods
        start = time.time()
        print('Starting run at time {}'.format(datetime.now().strftime("%H:%M:%S")))
        run_object.train_and_test_model()
        run_object.calculate_metrics()

        if run_object.feature_extraction is not False:
            from harness.feature_extraction_run_classes import FeatureExtractionRun
            feature_extractor = FeatureExtractionRun(base_run_instance=run_object)
            feature_extractor.feature_extraction_method(method=run_object.feature_extraction)

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
            row_of_results[Names.TEST_GROUP] = run_object.loo_dict["group_info"]
        print()
        print(row_of_results)
        print()

        # update leaderboard with new entry (row_of_results) and sort it based on run type
        leaderboard = leaderboard.append(row_of_results, ignore_index=True, sort=False)
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
        print(run_object.run_id)
        row_values = {Names.RUN_ID: run_object.run_id, Names.DATE: run_object.date_ran, Names.TIME: run_object.time_ran,
                      Names.SAMPLES_IN_TRAIN: run_object.metrics_dict[Names.SAMPLES_IN_TRAIN],
                      Names.SAMPLES_IN_TEST: run_object.metrics_dict[Names.SAMPLES_IN_TEST],
                      Names.MODEL_NAME: run_object.model_name, Names.MODEL_AUTHOR: run_object.model_author,
                      Names.MODEL_DESCRIPTION: run_object.model_description, Names.COLUMN_PREDICTED: run_object.col_to_predict,
                      Names.NUM_FEATURES_USED: run_object.metrics_dict[Names.NUM_FEATURES_USED],
                      Names.DATA_AND_SPLIT_DESCRIPTION: run_object.data_and_split_description, Names.NORMALIZED: run_object.normalize,
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

    def _output_run_files(self, run_object, output_path, output_data_csvs=True):
        if output_data_csvs:
            # using index_cols and prediction/ranking cols to only output subset of dataframe.
            # using unchanged_index_cols to get names of columns that were created in execute_run for later output.
            # thus what is output are the original input columns and not transformed input columns (e.g. if normalization is used)

            unchanged_index_cols = ["unchanged_{}".format(x) for x in run_object.index_cols]

            train_cols_to_output = unchanged_index_cols
            if run_object.run_type == Names.CLASSIFICATION:
                test_cols_to_output = unchanged_index_cols + [run_object.predictions_col, run_object.prob_predictions_col]
            elif run_object.run_type == Names.REGRESSION:
                test_cols_to_output = unchanged_index_cols + [run_object.predictions_col, run_object.residuals_col]
            else:
                raise ValueError("run_object.run_type must be {} or {}".format(Names.REGRESSION, Names.CLASSIFICATION))

            train_df_to_output = run_object.training_data[train_cols_to_output].copy()
            for col in unchanged_index_cols:
                train_df_to_output.rename(columns={col: col.rsplit("unchanged_")[1]}, inplace=True)
            train_df_to_output.to_csv('{}/{}'.format(output_path, 'training_data.csv'), index=False)

            test_df_to_output = run_object.testing_data_predictions[test_cols_to_output].copy()
            for col in unchanged_index_cols:
                test_df_to_output.rename(columns={col: col.rsplit("unchanged_")[1]}, inplace=True)
            test_df_to_output.to_csv('{}/{}'.format(output_path, 'testing_data.csv'), index=False)

            if run_object.was_untested_data_predicted is not False:
                pred_cols_to_output = test_cols_to_output + [run_object.rankings_col]
                prediction_data_to_output = run_object.untested_data_predictions[pred_cols_to_output].copy()
                for col in unchanged_index_cols:
                    prediction_data_to_output.rename(columns={col: col.rsplit("unchanged_")[1]}, inplace=True)
                prediction_data_to_output.to_csv('{}/{}'.format(output_path, 'predicted_data.csv'), index=False)
        if run_object.feature_extraction is not False:
            run_object.feature_importances.to_csv('{}/{}'.format(output_path, 'feature_importances.csv'), index=False)
            if run_object.feature_extraction == Names.SHAP_AUDIT:
                shap_path = os.path.join(output_path, 'SHAP')
                if not os.path.exists(shap_path):
                    os.makedirs(shap_path)
                dependence_path = os.path.join(shap_path, 'feature_dependence_plots')
                if not os.path.exists(dependence_path):
                    os.makedirs(dependence_path)
                run_object.shap_values.to_csv('{}/{}'.format(shap_path, 'shap_values.csv'), index=False)
                for name, plot in run_object.shap_plots_dict.items():
                    if "dependence_plot" in name:
                        plot.savefig(os.path.join(dependence_path, name), bbox_inches="tight")
                    else:
                        plot.savefig(os.path.join(shap_path, name), bbox_inches="tight")

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

        if run_object.normalization_scaler_object is not None:
            joblib.dump(run_object.normalization_scaler_object, os.path.join(output_path, "normalization_scaler_object.pkl"))

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
