import time
import pandas as pd
from math import sqrt
from test_harness.unique_id import get_id
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, r2_score



class Run:
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data):
        self.test_harness_model = test_harness_model
        self.model_description = test_harness_model.model_description
        self.stack_trace = test_harness_model.stack_trace
        self.training_data = training_data
        self.testing_data = testing_data
        self.data_and_split_description = data_and_split_description
        self.col_to_predict = col_to_predict
        self.feature_cols_to_use = feature_cols_to_use
        self.normalize = normalize
        self.feature_cols_to_normalize = feature_cols_to_normalize
        self.feature_extraction = feature_extraction
        self.predict_untested_data = predict_untested_data
        self.predictions_col = "{}_predictions".format(col_to_predict)
        self.run_id = get_id()
        if self.predict_untested_data is False:
            self.was_untested_data_predicted = False
        else:
            self.was_untested_data_predicted = True

    # def normalize_data(self):
    #     scaler = preprocessing.StandardScaler().fit(self.training_data[self.feature_cols_to_normalize])
    #     normalized_train = self.training_data.copy()
    #     normalized_train[self.feature_cols_to_normalize] = scaler.transform(
    #         normalized_train[self.feature_cols_to_normalize])
    #     normalized_test = self.testing_data.copy()
    #     normalized_test[self.feature_cols_to_normalize] = scaler.transform(
    #         normalized_test[self.feature_cols_to_normalize])
    #     self.training_data, self.testing_data = normalized_train.copy(), normalized_test.copy()


class ClassificationRun(Run):
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data):
        super().__init__(test_harness_model, training_data, testing_data, data_and_split_description,
                         col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                         feature_extraction, predict_untested_data)
        self.prob_predictions_col = "{}_prob_predictions".format(col_to_predict)


        # execution_id, run_id, auc, accuracy, model_description, col_to_predict, num_features_used,
        # data_and_split_description, normalize, num_features_normalized, feature_extraction,
        # was_untested_data_predicted, stack_trace, train_df, test_df_with_preds,
        # untested_df_with_preds = False

    def train_and_test(self):
        train_df = self.training_data.copy()
        test_df = self.testing_data.copy()

        # Training model
        training_start_time = time.time()
        self.test_harness_model._fit(train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("Classifier training time was: {}".format(time.time() - training_start_time)))

        # Testing model
        testing_start_time = time.time()
        test_df.loc[:, self.predictions_col] = self.test_harness_model._predict(test_df[self.feature_cols_to_use])
        test_df.loc[:, self.prob_predictions_col] = self.test_harness_model._predict_proba(test_df[self.feature_cols_to_use])
        print(("Classifier testing time was: {}".format(time.time() - testing_start_time)))
        # Saving predictions for calculating metrics later
        self.testing_data_predictions = test_df.copy()

        # Predicting values for untested dataset if applicable
        if self.predict_untested_data is not False:
            untested_df = self.predict_untested_data.copy()
            prediction_start_time = time.time()
            untested_df.loc[:, self.predictions_col] = self.test_harness_model._predict(untested_df[self.feature_cols_to_use])
            untested_df.loc[:, self.prob_predictions_col] = self.test_harness_model._predict_proba(untested_df[self.feature_cols_to_use])
            print(("Classifier prediction time of untested data was: {}".format(time.time() - prediction_start_time)))
            untested_df.sort_values(self.predictions_col, inplace=True, ascending=False)
            # Saving untested predictions
            self.untested_data_predictions = untested_df.copy()
        else:
            self.untested_data_predictions = None

    def calculate_metrics(self):
        self.num_features_used = len(self.feature_cols_to_use)
        self.num_features_normalized = len(self.feature_cols_to_normalize)
        self.num_datapoints = len(self.testing_data_predictions)
        total_equal = sum(self.testing_data_predictions[self.col_to_predict] == self.testing_data_predictions[self.predictions_col])
        self.percent_accuracy = float(total_equal) / float(self.num_datapoints)
        self.auc_score = roc_auc_score(self.testing_data_predictions[self.col_to_predict], self.testing_data_predictions[self.prob_predictions_col])


class RegressionRun(Run):
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data):
        super().__init__(test_harness_model, training_data, testing_data, data_and_split_description,
                         col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                         feature_extraction, predict_untested_data)
        self.residuals_col = "{}_residuals".format(col_to_predict)


    def train_and_test(self):
        train_df = self.training_data.copy()
        test_df = self.testing_data.copy()

        # Training model
        training_start_time = time.time()
        self.test_harness_model._fit(train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("Regressor training time was: {}".format(time.time() - training_start_time)))

        # Testing model
        testing_start_time = time.time()
        test_df.loc[:, self.predictions_col] = self.test_harness_model._predict(test_df[self.feature_cols_to_use])
        test_df[self.residuals_col] = test_df[self.col_to_predict] - test_df[self.predictions_col]
        print(("Regressor testing time was: {}".format(time.time() - testing_start_time)))
        # Saving predictions for calculating metrics later
        self.testing_data_predictions = test_df.copy()

        # Predicting values for untested dataset if applicable
        if self.predict_untested_data is not False:
            untested_df = self.predict_untested_data.copy()
            prediction_start_time = time.time()
            untested_df.loc[:, self.predictions_col] = self.test_harness_model._predict(untested_df[self.feature_cols_to_use])
            print(("Regressor prediction time (untested data) was: {}".format(time.time() - prediction_start_time)))
            untested_df.sort_values(self.predictions_col, inplace=True, ascending=False)
            # Saving untested predictions
            self.untested_data_predictions = untested_df.copy()
        else:
            self.untested_data_predictions = None

    def calculate_metrics(self):
        self.num_features_used = len(self.feature_cols_to_use)
        self.num_features_normalized = len(self.feature_cols_to_normalize)
        self.num_datapoints = len(self.testing_data_predictions)
        self.rmse = sqrt(mean_squared_error(self.testing_data_predictions[self.col_to_predict], self.testing_data_predictions[self.predictions_col]))
        self.r_squared = r2_score(self.testing_data_predictions[self.col_to_predict], self.testing_data_predictions[self.predictions_col])