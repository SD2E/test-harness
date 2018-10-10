import os
import json
import time
import pandas as pd
from unique_id import get_id
from sklearn import preprocessing

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
            print('total time elapsed for this model = {}'.format(end - start))
            print()
            self._finished_models.append((model_runner, results))

    def run_model_on_grouping_splits(self, function_that_returns_model_runner, all_data_df, grouping_df, col_to_predict,
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

    def run_model_general(self, model_runner_instance, train, test, is_this_sequence_cnn=False,
                          normalize=False, feature_cols_to_normalize=None,
                          get_pimportances=False, performance_output_path=None, features_output_path=None):

        results = pd.DataFrame()
        features = None
        train_split = train.copy()
        test_split = test.copy()
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

        if is_this_sequence_cnn is True:
            this_run_results = model_runner_instance.run_model()
        else:
            this_run_results = model_runner_instance.run_model(train_split, test_split)
        this_run_results['num_proteins_in_train_set'] = len(train_split)
        this_run_results['num_proteins_in_test_set'] = len(test_split)
        print(this_run_results)
        results = pd.concat([results, this_run_results])

        if get_pimportances is True:
            this_run_perms = model_runner_instance.permutation_importances
            if isinstance(this_run_perms, pd.DataFrame):
                if features is None:
                    features = this_run_perms
                else:
                    features = pd.merge(features, this_run_perms, on='Feature')
        print()
        splits_results = results.sort_values('AUC Score', ascending=False)
        print(splits_results)
        print()
        print(features)
        if performance_output_path is not None:
            print(performance_output_path)
            splits_results.to_csv(performance_output_path, index=False)
        if features is not None and features_output_path is not None:
            features.to_csv(features_output_path, index=False)

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
