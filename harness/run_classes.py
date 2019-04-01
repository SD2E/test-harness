from datetime import datetime
from math import sqrt
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, r2_score

from harness.unique_id import get_id
from harness.utils.names import Names
from harness.test_harness_models_abstract_classes import ClassificationModel, RegressionModel
from harness.utils.object_type_modifiers_and_checkers import is_list_of_strings


class _BaseRun:
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, index_cols, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data=False, sparse_cols_to_use=None, loo_dict=False):
        if isinstance(test_harness_model, ClassificationModel):
            self.run_type = Names.CLASSIFICATION
            self.prob_predictions_col = "{}_prob_predictions".format(col_to_predict)
        elif isinstance(test_harness_model, RegressionModel):
            self.run_type = Names.REGRESSION
            self.residuals_col = "{}_residuals".format(col_to_predict)
        else:
            raise TypeError("test_harness_model must be a ClassificationModel or a RegressionModel")
        self.test_harness_model = test_harness_model
        self.model_name = test_harness_model.model_name
        self.model_author = test_harness_model.model_author
        self.model_description = test_harness_model.model_description
        self.model_stack_trace = test_harness_model.stack_trace
        self.training_data = training_data
        self.testing_data = testing_data
        self.data_and_split_description = data_and_split_description
        self.col_to_predict = col_to_predict
        self.feature_cols_to_use = feature_cols_to_use
        self.index_cols = index_cols
        self.normalize = normalize
        self.feature_cols_to_normalize = feature_cols_to_normalize
        self.feature_extraction = feature_extraction
        self.predict_untested_data = predict_untested_data
        self.sparse_cols_to_use = sparse_cols_to_use
        self.predictions_col = "{}_predictions".format(col_to_predict)
        self.rankings_col = "{}_rankings".format(col_to_predict)
        self.run_id = get_id()
        self.loo_dict = loo_dict
        if self.predict_untested_data is False:
            self.was_untested_data_predicted = False
        else:
            self.was_untested_data_predicted = True
        self.date_ran = datetime.now().strftime("%Y-%m-%d")
        self.time_ran = datetime.now().strftime("%H:%M:%S")
        self.metrics_dict = {}
        self.normalization_scaler_object = None

    def _normalize_dataframes(self):
        warnings.simplefilter('ignore', DataConversionWarning)

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

        # saving fitted scaler as an instance variable. In test_harness_class.py this variable will be saved via joblib.
        self.normalization_scaler_object = scaler

        train_df[self.feature_cols_to_normalize] = scaler.transform(train_df[self.feature_cols_to_normalize])
        test_df[self.feature_cols_to_normalize] = scaler.transform(test_df[self.feature_cols_to_normalize])
        self.training_data = train_df.copy()
        self.testing_data = test_df.copy()

        # Normalizing untested dataset if applicable
        if self.predict_untested_data is not False:
            untested_df = self.predict_untested_data.copy()
            untested_df[self.feature_cols_to_normalize] = scaler.transform(untested_df[self.feature_cols_to_normalize])
            self.predict_untested_data = untested_df.copy()

    # TODO: Put in a check to never normalize the sparse data category
    def _add_sparse_cols(self):
        assert is_list_of_strings(self.sparse_cols_to_use), \
            "self.sparse_cols_to_use must be a string or a list of strings when the _add_sparse_cols method is called."

        # obtain all the sparse column values present in training_data, testing_data, and untested_data (if applicable)
        for sparse_col in self.sparse_cols_to_use:
            train_vals = set(self.training_data[sparse_col].unique())
            test_vals = set(self.testing_data[sparse_col].unique())
            if self.was_untested_data_predicted:
                untested_vals = set(self.predict_untested_data[sparse_col].unique())
            else:
                untested_vals = set()
            all_vals_for_this_sparse_col = set().union(train_vals, test_vals, untested_vals)
            print(train_vals)
            print(test_vals)
            print(untested_vals)
            print(all_vals_for_this_sparse_col)

            # update self.feature_cols_to_use
            self.feature_cols_to_use.remove(sparse_col)
            self.feature_cols_to_use.extend(['{}_{}'.format(sparse_col, val) for val in all_vals_for_this_sparse_col])

            # update training data:
            self.training_data = pd.get_dummies(self.training_data, columns=[sparse_col])
            for val in all_vals_for_this_sparse_col.difference(train_vals):
                self.training_data['{}_{}'.format(sparse_col, val)] = 0

            # update testing data:
            self.testing_data = pd.get_dummies(self.testing_data, columns=[sparse_col])
            for val in all_vals_for_this_sparse_col.difference(test_vals):
                self.testing_data['{}_{}'.format(sparse_col, val)] = 0

            # update untested data:
            if self.was_untested_data_predicted:
                self.predict_untested_data = pd.get_dummies(self.predict_untested_data, columns=[sparse_col])
                for val in all_vals_for_this_sparse_col.difference(untested_vals):
                    self.predict_untested_data['{}_{}'.format(sparse_col, val)] = 0

    def train_and_test_model(self):
        if self.normalize:
            self._normalize_dataframes()

        if self.sparse_cols_to_use:
            self._add_sparse_cols()

        train_df = self.training_data.copy()
        test_df = self.testing_data.copy()

        # Training model
        print("Starting {} training...".format(self.run_type))
        training_start_time = time.time()
        self.test_harness_model._fit(train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("Training time was: {0:.2f} seconds".format(time.time() - training_start_time)))

        # Testing model
        testing_start_time = time.time()
        test_df.loc[:, self.predictions_col] = self.test_harness_model._predict(test_df[self.feature_cols_to_use])
        if self.run_type == Names.CLASSIFICATION:
            test_df.loc[:, self.prob_predictions_col] = self.test_harness_model._predict_proba(test_df[self.feature_cols_to_use])
        elif self.run_type == Names.REGRESSION:
            test_df[self.residuals_col] = test_df[self.col_to_predict] - test_df[self.predictions_col]
        else:
            raise ValueError(
                "run_type must be '{}' or '{}'".format(Names.CLASSIFICATION, Names.REGRESSION))
        print(("Testing time was: {0:.2f} seconds".format(time.time() - testing_start_time)))
        # Saving predictions for calculating metrics later
        self.testing_data_predictions = test_df.copy()

        # Predicting values for untested dataset if applicable
        if self.predict_untested_data is not False:
            untested_df = self.predict_untested_data.copy()
            prediction_start_time = time.time()

            untested_df.loc[:, self.predictions_col] = self.test_harness_model._predict(untested_df[self.feature_cols_to_use])
            if self.run_type == Names.CLASSIFICATION:
                untested_df.loc[:, self.prob_predictions_col] = \
                    self.test_harness_model._predict_proba(untested_df[self.feature_cols_to_use])

            # IDEA: remove all columns except for self.index_cols and self.predictions_col. This is already done in test_harness_class.py,
            # IDEA: but if it's done here the extra columns wouldn't have to be stored in the run_object either.

            # creating rankings column based on the predictions. Rankings assume that a higher score is more desirable
            if self.run_type == Names.REGRESSION:
                untested_df[self.rankings_col] = untested_df.sort_values(by=[self.predictions_col], ascending=False)[
                                                     self.predictions_col].index + 1
            elif self.run_type == Names.CLASSIFICATION:
                untested_df[self.rankings_col] = untested_df.sort_values(by=[self.predictions_col, self.prob_predictions_col],
                                                                         ascending=[False, False])[self.predictions_col].index + 1
            else:
                raise ValueError("self.run_type must be {} or {}".format(Names.REGRESSION, Names.CLASSIFICATION))

            print(("Prediction time of untested data was: {}".format(time.time() - prediction_start_time)))
            untested_df.sort_values(self.predictions_col, inplace=True, ascending=False)
            # Saving untested predictions
            self.untested_data_predictions = untested_df.copy()
        else:
            self.untested_data_predictions = None

    def calculate_metrics(self):
        self.metrics_dict[Names.NUM_FEATURES_USED] = len(self.feature_cols_to_use)
        if self.feature_cols_to_normalize:
            self.metrics_dict[Names.NUM_FEATURES_NORMALIZED] = len(self.feature_cols_to_normalize)
        else:
            self.metrics_dict[Names.NUM_FEATURES_NORMALIZED] = 0
        self.metrics_dict[Names.SAMPLES_IN_TRAIN] = len(self.training_data)
        self.metrics_dict[Names.SAMPLES_IN_TEST] = len(self.testing_data_predictions)

        if self.run_type == Names.CLASSIFICATION:
            try:
                self.metrics_dict[Names.AUC_SCORE] = roc_auc_score(self.testing_data_predictions[self.col_to_predict],
                                                                   self.testing_data_predictions[self.prob_predictions_col])
            except ValueError:
                self.metrics_dict[Names.AUC_SCORE] = np.NaN

            try:
                self.metrics_dict[Names.AVERAGE_PRECISION] = \
                    average_precision_score(self.testing_data_predictions[self.col_to_predict],
                                            self.testing_data_predictions[self.prob_predictions_col])
            except ValueError:
                self.metrics_dict[Names.AVERAGE_PRECISION] = np.NaN

            self.metrics_dict[Names.ACCURACY] = accuracy_score(self.testing_data_predictions[self.col_to_predict],
                                                               self.testing_data_predictions[self.predictions_col])
            self.metrics_dict[Names.BALANCED_ACCURACY] = balanced_accuracy_score(self.testing_data_predictions[self.col_to_predict],
                                                                                 self.testing_data_predictions[self.predictions_col])
            self.metrics_dict[Names.F1_SCORE] = f1_score(self.testing_data_predictions[self.col_to_predict],
                                                         self.testing_data_predictions[self.predictions_col])
            self.metrics_dict[Names.PRECISION] = precision_score(self.testing_data_predictions[self.col_to_predict],
                                                                 self.testing_data_predictions[self.predictions_col])
            self.metrics_dict[Names.RECALL] = recall_score(self.testing_data_predictions[self.col_to_predict],
                                                           self.testing_data_predictions[self.predictions_col])
        elif self.run_type == Names.REGRESSION:
            self.metrics_dict[Names.RMSE] = sqrt(
                mean_squared_error(self.testing_data_predictions[self.col_to_predict], self.testing_data_predictions[self.predictions_col]))
            self.metrics_dict[Names.R_SQUARED] = r2_score(self.testing_data_predictions[self.col_to_predict],
                                                          self.testing_data_predictions[self.predictions_col])
        else:
            raise TypeError("self.run_type must equal '{}' or '{}'".format(Names.CLASSIFICATION, Names.REGRESSION))
