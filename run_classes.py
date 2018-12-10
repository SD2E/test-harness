import time
import eli5
import rfpimp
import pandas as pd
from math import sqrt
from test_harness.unique_id import get_id
from datetime import datetime
from sklearn import preprocessing
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, r2_score


class BaseRun:
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data=False, loo_dict=False):
        self.test_harness_model = test_harness_model
        self.model_description = test_harness_model.model_description
        self.model_stack_trace = test_harness_model.stack_trace
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
        self.loo_dict = loo_dict
        if self.predict_untested_data is False:
            self.was_untested_data_predicted = False
        else:
            self.was_untested_data_predicted = True
        self.date_ran = datetime.now().strftime("%Y-%m-%d")
        self.time_ran = datetime.now().strftime("%H:%M:%S")

    def _normalize_dataframes(self):
        if self.feature_cols_to_normalize is None:
            raise ValueError("feature_cols_to_normalize must be a list of column names if you are trying to normalize the data.")

        train_df = self.training_data.copy()
        test_df = self.testing_data.copy()

        print("Normalizing training and testing splits...")
        if (self.normalize is True) or (self.normalize == "StandardScaler"):
            scaler = preprocessing.StandardScaler().fit(train_df[self.feature_cols_to_normalize])
        elif self.normalize == "MinMax":
            raise ValueError("MinMax normalization hasn't been added yet")
        else:
            raise ValueError("normalize must have a value of True, 'StandardScaler', or 'MinMax'")

        train_df[self.feature_cols_to_normalize] = scaler.transform(train_df[self.feature_cols_to_normalize])
        test_df[self.feature_cols_to_normalize] = scaler.transform(test_df[self.feature_cols_to_normalize])
        self.training_data = train_df.copy()
        self.testing_data = test_df.copy()

        # Normalizing untested dataset if applicable
        if self.predict_untested_data is not False:
            untested_df = self.predict_untested_data.copy()
            untested_df[self.feature_cols_to_normalize] = scaler.transform(untested_df[self.feature_cols_to_normalize])
            self.predict_untested_data = untested_df.copy()

    # TODO: add different options for eli5.sklearn.permutation_importance (current usage) and eli5.permutation_importance
    def feature_extraction_method(self, method="eli5_permutation"):
        print("Starting Feature Extraction...")
        start_time = time.time()

        if method == True:
            method = "eli5_permutation"

        if method == "eli5_permutation":
            pi_object = PermutationImportance(self.test_harness_model.model)
            pi_object.fit(self.testing_data[self.feature_cols_to_use], self.testing_data[self.col_to_predict])
            feature_importances_df = pd.DataFrame()
            feature_importances_df["Feature"] = self.feature_cols_to_use
            feature_importances_df["Importance"] = pi_object.feature_importances_
            feature_importances_df["Importance_Std"] = pi_object.feature_importances_std_
            feature_importances_df.sort_values(by='Importance', inplace=True, ascending=False)
            self.feature_importances = feature_importances_df.copy()
        elif method == "rfpimp_permutation":
            pis = rfpimp.importances(self.test_harness_model.model, self.testing_data[self.feature_cols_to_use], self.testing_data[self.col_to_predict])
            pis['Feature'] = pis.index
            pis.reset_index(inplace=True, drop=True)
            pis = pis[['Feature', 'Importance']]
            pis.sort_values(by='Importance', inplace=True, ascending=False)
            self.feature_importances = pis.copy()
        elif method == "sklearn_rf_default":
            pass      # TODO

        print(("Feature Extraction time with method {0} was: {1:.2f} seconds".format(method, time.time() - start_time)))



class ClassificationRun(BaseRun):
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data=False, loo_dict=False):
        super().__init__(test_harness_model, training_data, testing_data, data_and_split_description,
                         col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                         feature_extraction, predict_untested_data, loo_dict)
        self.prob_predictions_col = "{}_prob_predictions".format(col_to_predict)

    def train_and_test_model(self):
        if self.normalize is not False:
            self._normalize_dataframes()

        train_df = self.training_data.copy()
        test_df = self.testing_data.copy()

        # Training model
        print("Starting Classifier training...")
        training_start_time = time.time()
        self.test_harness_model._fit(train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("Classifier training time was: {0:.2f} seconds".format(time.time() - training_start_time)))

        # Testing model
        testing_start_time = time.time()
        test_df.loc[:, self.predictions_col] = self.test_harness_model._predict(test_df[self.feature_cols_to_use])
        test_df.loc[:, self.prob_predictions_col] = self.test_harness_model._predict_proba(test_df[self.feature_cols_to_use])
        print(("Classifier testing time was: {0:.2f} seconds".format(time.time() - testing_start_time)))
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
        self.number_of_test_datapoints = len(self.testing_data_predictions)
        total_equal = sum(self.testing_data_predictions[self.col_to_predict] == self.testing_data_predictions[self.predictions_col])
        self.percent_accuracy = float(total_equal) / float(self.number_of_test_datapoints)
        self.auc_score = roc_auc_score(self.testing_data_predictions[self.col_to_predict],
                                       self.testing_data_predictions[self.prob_predictions_col])


class RegressionRun(BaseRun):
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data=False, loo_dict=False):
        super().__init__(test_harness_model, training_data, testing_data, data_and_split_description,
                         col_to_predict, feature_cols_to_use, normalize, feature_cols_to_normalize,
                         feature_extraction, predict_untested_data, loo_dict)
        self.residuals_col = "{}_residuals".format(col_to_predict)

    def train_and_test_model(self):
        if self.normalize is not False:
            self._normalize_dataframes()

        train_df = self.training_data.copy()
        test_df = self.testing_data.copy()

        # Training model
        print("Starting Regressor training...")
        training_start_time = time.time()
        self.test_harness_model._fit(train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("Regressor training time was: {0:.2f} seconds".format(time.time() - training_start_time)))

        # Testing model
        testing_start_time = time.time()
        test_df.loc[:, self.predictions_col] = self.test_harness_model._predict(test_df[self.feature_cols_to_use])
        test_df[self.residuals_col] = test_df[self.col_to_predict] - test_df[self.predictions_col]
        print(("Regressor testing time was: {0:.2f} seconds".format(time.time() - testing_start_time)))
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
        self.number_of_test_datapoints = len(self.testing_data_predictions)
        self.rmse = sqrt(
            mean_squared_error(self.testing_data_predictions[self.col_to_predict], self.testing_data_predictions[self.predictions_col]))
        self.r_squared = r2_score(self.testing_data_predictions[self.col_to_predict], self.testing_data_predictions[self.predictions_col])