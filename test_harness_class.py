import os
import json
import time
import pandas as pd
from unique_id import get_id

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
# CSS classes applied to the Pandas Dataframes when written as HTML
css_classes = ["table-bordered", "table-striped", "table-compact"]


class TestHarness:
    def __init__(self, output_path=os.path.dirname(os.path.realpath(__file__))):
        self.output_path = output_path
        self.model_runners = []
        self._finished_models = []
        self.initialize_leaderboards()

    def add_model_runner(self, model_runner_instance):
        self.model_runners.append(model_runner_instance)

    def remove_model_runner(self, model_runner_instance):
        self.model_runners.remove(model_runner_instance)

    def initialize_leaderboards(self):
        self.class_leaderboard_cols = ['Run ID', 'AUC Score', 'Classification Accuracy',
                                       'Model Description', 'Number Of Features Used', 'Column Predicted',
                                       'Data Set Description', 'Train/Test Split Description',
                                       'Topology Specific or General?']

        self.reg_leaderboard_cols = ['Run ID', 'RMSE', 'Percent Error', 'R Squared',
                                     'Model Description', 'Number Of Features Used', 'Column Predicted',
                                     'Data Set Description', 'Train/Test Split Description',
                                     'Topology Specific or General?']

        cc_leaderboard_name = 'comparable_classification_leaderboard'
        html_path = os.path.join(self.output_path, "{}.html".format(cc_leaderboard_name))
        try:
            cc_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            cc_leaderboard = pd.DataFrame(columns=self.class_leaderboard_cols)
            cc_leaderboard.to_html(
                html_path,
                index=False,
                classes='comparable_classification')
        self.comparable_classification_leaderboard = cc_leaderboard.copy()

        gc_leaderboard_name = 'general_classification_leaderboard'
        html_path = os.path.join(self.output_path, "{}.html".format(gc_leaderboard_name))
        try:
            gc_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            gc_leaderboard = pd.DataFrame(columns=self.class_leaderboard_cols)
            gc_leaderboard.to_html(
                html_path,
                index=False,
                classes='general_classification')
        self.general_classification_leaderboard = gc_leaderboard.copy()

        cr_leaderboard_name = 'comparable_regression_leaderboard'
        html_path = os.path.join(self.output_path, "{}.html".format(cr_leaderboard_name))
        try:
            cr_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            cr_leaderboard = pd.DataFrame(columns=self.reg_leaderboard_cols)
            cr_leaderboard.to_html(
                html_path,
                index=False,
                classes='comparable_regression')
        self.comparable_regression_leaderboard = cr_leaderboard.copy()

        gr_leaderboard_name = 'general_regression_leaderboard'
        html_path = os.path.join(self.output_path, "{}.html".format(gr_leaderboard_name))
        try:
            gr_leaderboard = pd.read_html(html_path)[0]
        except (IOError, ValueError):
            gr_leaderboard = pd.DataFrame(columns=self.reg_leaderboard_cols)
            gr_leaderboard.to_html(
                html_path,
                index=False,
                classes='general_regression')
        self.general_regression_leaderboard = gr_leaderboard.copy()

    def run_models(self):
        for model_runner in self.model_runners:
            print()
            print('Starting model with description: {}'.format(model_runner.model_description))
            start = time.time()
            print('model started at {}'.format(start))
            results = model_runner.run_model()
            end = time.time()
            print('model finished at {}'.format(end))
            print('total time elapsed for this model = {}'.format(end-start))
            print()
            self._finished_models.append((model_runner, results))

    def run_models_on_splits_by_columns(self, cols=['topology', 'library'], performance_output_path=None,
                                       features_output_path=None):
        for model_runner in self.model_runners:
            split_results, split_features = model_runner.splits_by_columns(cols=cols)
            split_results = split_results.sort_values('R Squared', ascending=False)
            print(split_results)
            print(split_features)
            if performance_output_path is not None:
                split_results.to_csv(performance_output_path, index=False)
            if features_output_path is not None:
                split_features.to_csv(features_output_path, index=False)

    def run_models_on_custom_splits(self, grouping_df, performance_output_path=None, features_output_path=None,
                                    normalize=True, get_pimportances=True):
        for model_runner in self.model_runners:
            split_results, split_features = model_runner.custom_splits(grouping_df, normalize=normalize,
                                                                       get_pimportances=get_pimportances)
            split_results = split_results.sort_values('R Squared', ascending=False)
            print(split_results)
            print(split_features)
            if performance_output_path is not None:
                split_results.to_csv(performance_output_path, index=False)
            if split_features is not None and features_output_path is not None:
                split_features.to_csv(features_output_path, index=False)

    def run_test_harness(self):
        for x in self._finished_models:
            model_runner_instance = x[0]
            model_runner_results = x[1]
            # print model_runner_instance.training_data
            # print model_runner_results
            # print model_runner_instance.type
            if model_runner_instance.type == 'classification':
                if model_runner_instance.default_data_set_used is True:
                    run_id = get_id()
                    model_runner_results['Run ID'] = run_id
                    self.comparable_classification_leaderboard = self.comparable_classification_leaderboard.append(
                        model_runner_results, ignore_index=True)
                    self.comparable_classification_leaderboard.sort_values('AUC Score', inplace=True, ascending=False)
                    self.comparable_classification_leaderboard.reset_index(inplace=True, drop=True)
                    html_path = os.path.join(self.output_path, "comparable_classification_leaderboard.html")
                    self.comparable_classification_leaderboard.to_html(
                        html_path,
                        index=False,
                        classes='comparable_classification')
                elif model_runner_instance.default_data_set_used is False:
                    run_id = get_id()
                    model_runner_results['Run ID'] = run_id
                    self.general_classification_leaderboard = self.general_classification_leaderboard.append(
                        model_runner_results, ignore_index=True)
                    self.general_classification_leaderboard.sort_values('AUC Score', inplace=True, ascending=False)
                    self.general_classification_leaderboard.reset_index(inplace=True, drop=True)
                    html_path = os.path.join(self.output_path, "general_classification_leaderboard.html")
                    self.general_classification_leaderboard.to_html(
                        html_path,
                        index=False,
                        classes='general_classification')
            elif model_runner_instance.type == 'regression':
                if model_runner_instance.default_data_set_used is True:
                    run_id = get_id()
                    model_runner_results['Run ID'] = run_id
                    self.comparable_regression_leaderboard = self.comparable_regression_leaderboard.append(
                        model_runner_results, ignore_index=True)
                    self.comparable_regression_leaderboard.sort_values('RMSE', inplace=True,
                                                                       ascending=True)
                    self.comparable_regression_leaderboard.reset_index(inplace=True, drop=True)
                    html_path = os.path.join(self.output_path, "comparable_regression_leaderboard.html")
                    self.comparable_regression_leaderboard.to_html(
                        html_path,
                        index=False,
                        classes='comparable_regression')
                elif model_runner_instance.default_data_set_used is False:
                    run_id = get_id()
                    model_runner_results['Run ID'] = run_id
                    self.general_regression_leaderboard = self.general_regression_leaderboard.append(
                        model_runner_results, ignore_index=True)
                    self.general_regression_leaderboard.sort_values('RMSE', inplace=True, ascending=True)
                    self.general_regression_leaderboard.reset_index(inplace=True, drop=True)
                    html_path = os.path.join(self.output_path, "general_regression_leaderboard.html")
                    self.general_regression_leaderboard.to_html(
                        html_path,
                        index=False,
                        classes='general_regression')


            else:
                raise ValueError()

            training_data_to_save = model_runner_instance.training_data.copy()
            testing_data_to_save = model_runner_instance.test_predictions_df.copy()
            if model_runner_instance.predict_untested is not False:
                prediction_data_to_save = model_runner_instance.predict_untested.copy()

            top_folder_path = os.path.join(self.output_path, 'model_outputs')
            sub_folder_path = '{}/{}'.format(top_folder_path, run_id)
            os.makedirs(sub_folder_path, exist_ok=True)
            training_data_to_save.to_csv('{}/{}'.format(sub_folder_path, 'training_data.csv'), index=False)
            testing_data_to_save.to_csv('{}/{}'.format(sub_folder_path, 'testing_data.csv'), index=False)
            if model_runner_instance.predict_untested is not False:
                prediction_data_to_save.to_csv('{}/{}'.format(sub_folder_path, 'predicted_data.csv'), index=False)
            test_file_name = os.path.join(sub_folder_path, 'model_information.txt')
            with open(test_file_name, "w") as f:
                f.write("Feature columns used by model: \n")
                json.dump(model_runner_instance.feature_cols_to_use, f)
                f.write("\n\n\n")

                f.write("Model Instantiation Trace:\n")
                for i, t in enumerate(model_runner_instance.stack_trace):
                    f.write(" Level {}\n".format(i))
                    path, line, func = t[1:4]
                    f.write(' - Path: ' + path + '\n')
                    f.write(' - Line: ' + str(line) + ',  Function: ' + str(func) + '\n')
                    f.write("\n")
