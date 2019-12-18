from datetime import datetime
from math import sqrt
import time
import warnings

import numpy as np
import pandas as pd
from copy import copy, deepcopy
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, r2_score

from harness.unique_id import get_id
from harness.utils.names import Names
from harness.test_harness_models_abstract_classes import ClassificationModel, RegressionModel
from harness.utils.object_type_modifiers_and_checkers import is_list_of_strings

'''
NOTE: If a class variable is going to be modified (e.g. feature_cols_to_use is modified by sparse col functionality),
then you must make sure that a COPY of the variable is passed in! Otherwise the original variable will be modified too, leading to issues.
'''


class _BaseRun:
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, index_cols, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data=False, sparse_cols_to_use=None, loo_dict=False,
                 interpret_complex_model=False, meta_model=False):
        if isinstance(test_harness_model, ClassificationModel):
            self.run_type = Names.CLASSIFICATION
            self.prob_predictions_col = "{}_prob_predictions".format(
                col_to_predict)
            unique_train_classes = set(training_data[col_to_predict].unique())
            unique_test_classes = set(testing_data[col_to_predict].unique())
            if unique_train_classes != unique_test_classes:
                warnings.warn("The unique classes in the training_data do not match those in the testing_data. "
                              "Perhaps you should stratify your train/test split based on your classes (col_to_predict)", Warning)
            num_classes = len(unique_train_classes)
            if num_classes > 2:
                self.multiclass = True
            else:
                self.multiclass = False
            self.num_classes = num_classes
        elif isinstance(test_harness_model, RegressionModel):
            self.run_type = Names.REGRESSION
            self.residuals_col = "{}_residuals".format(col_to_predict)
        else:
            raise TypeError(
                "test_harness_model must be a ClassificationModel or a RegressionModel")
        self.test_harness_model = test_harness_model
        self.model_name = test_harness_model.model_name
        self.model_author = test_harness_model.model_author
        self.model_description = test_harness_model.model_description
        self.model_stack_trace = test_harness_model.stack_trace
        self.training_data = training_data.copy()
        self.testing_data = testing_data.copy()
        self.data_and_split_description = data_and_split_description
        self.col_to_predict = col_to_predict
        self.feature_cols_to_use = copy(feature_cols_to_use)
        self.index_cols = copy(index_cols)
        self.normalize = normalize
        self.feature_cols_to_normalize = copy(feature_cols_to_normalize)
        self.feature_extraction = feature_extraction
        self.predict_untested_data = predict_untested_data
        self.sparse_cols_to_use = copy(sparse_cols_to_use)
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

        # model on model
        self.interpret_complex_model = interpret_complex_model
        self.model_interpretation_img = None

        # Meta-model like MAML
        self.meta_model = meta_model

    def _normalize_dataframes(self):
        warnings.simplefilter('ignore', DataConversionWarning)

        if self.feature_cols_to_normalize is None:
            raise ValueError(
                "feature_cols_to_normalize must be a list of column names if you are trying to normalize the data.")

        train_df = self.training_data.copy()
        test_df = self.testing_data.copy()

        print("Normalizing training and testing splits...")
        if (self.normalize is True) or (self.normalize == "StandardScaler"):
            scaler = preprocessing.StandardScaler().fit(
                train_df[self.feature_cols_to_normalize])
        elif self.normalize == "MinMax":
            raise ValueError("MinMax normalization hasn't been added yet")
        else:
            raise ValueError(
                "normalize must have a value of True, 'StandardScaler', or 'MinMax'")

        # saving fitted scaler as an instance variable. In test_harness_class.py this variable will be saved via joblib.
        self.normalization_scaler_object = scaler

        train_df[self.feature_cols_to_normalize] = scaler.transform(
            train_df[self.feature_cols_to_normalize])
        test_df[self.feature_cols_to_normalize] = scaler.transform(
            test_df[self.feature_cols_to_normalize])
        self.training_data = train_df.copy()
        self.testing_data = test_df.copy()

        # Normalizing untested dataset if applicable
        if self.predict_untested_data is not False:
            untested_df = self.predict_untested_data.copy()
            untested_df[self.feature_cols_to_normalize] = scaler.transform(
                untested_df[self.feature_cols_to_normalize])
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
                untested_vals = set(
                    self.predict_untested_data[sparse_col].unique())
            else:
                untested_vals = set()
            all_vals_for_this_sparse_col = set().union(
                train_vals, test_vals, untested_vals)

            # update self.feature_cols_to_use
            self.feature_cols_to_use.remove(sparse_col)
            self.feature_cols_to_use.extend(
                ['{}_{}'.format(sparse_col, val) for val in all_vals_for_this_sparse_col])

            # update training data:
            self.training_data = pd.get_dummies(
                self.training_data, columns=[sparse_col])
            for val in all_vals_for_this_sparse_col.difference(train_vals):
                self.training_data['{}_{}'.format(sparse_col, val)] = 0

            # update testing data:
            self.testing_data = pd.get_dummies(
                self.testing_data, columns=[sparse_col])
            for val in all_vals_for_this_sparse_col.difference(test_vals):
                self.testing_data['{}_{}'.format(sparse_col, val)] = 0

            # update untested data:
            if self.was_untested_data_predicted:
                self.predict_untested_data = pd.get_dummies(
                    self.predict_untested_data, columns=[sparse_col])
                for val in all_vals_for_this_sparse_col.difference(untested_vals):
                    self.predict_untested_data['{}_{}'.format(
                        sparse_col, val)] = 0

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
        self.test_harness_model._fit(
            train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("Training time was: {0:.2f} seconds".format(
            time.time() - training_start_time)))

        # Testing model
        testing_start_time = time.time()
        if self.meta_model:
            test_df.loc[:, self.predictions_col] = self.test_harness_model._predict(
                test_df)
        else:
            test_df.loc[:, self.predictions_col] = self.test_harness_model._predict(
                test_df[self.feature_cols_to_use])

        if self.run_type == Names.CLASSIFICATION:
            # _predict_proba currently returns the probability of class = 1
            test_df.loc[:, self.prob_predictions_col] = self.test_harness_model._predict_proba(
                test_df[self.feature_cols_to_use])
        elif self.run_type == Names.REGRESSION:
            test_df[self.residuals_col] = test_df[self.col_to_predict] - \
                test_df[self.predictions_col]
        else:
            raise ValueError(
                "run_type must be '{}' or '{}'".format(Names.CLASSIFICATION, Names.REGRESSION))
        print(("Testing time was: {0:.2f} seconds".format(
            time.time() - testing_start_time)))
        # Saving predictions for calculating metrics later
        self.testing_data_predictions = test_df.copy()

        # Predicting values for untested dataset if applicable
        if self.predict_untested_data is not False:
            untested_df = self.predict_untested_data.copy()
            prediction_start_time = time.time()

            untested_df.loc[:, self.predictions_col] = self.test_harness_model._predict(
                untested_df[self.feature_cols_to_use])
            if self.run_type == Names.CLASSIFICATION:
                # _predict_proba currently returns the probability of class = 1
                untested_df.loc[:, self.prob_predictions_col] = self.test_harness_model._predict_proba(
                    untested_df[self.feature_cols_to_use])

            # IDEA: remove all columns except for self.index_cols and self.predictions_col. This is already done in test_harness_class.py,
            # IDEA: but if it's done here the extra columns wouldn't have to be stored in the run_object either.

            # creating rankings column based on the predictions. Rankings assume that a higher score is more desirable
            if self.run_type == Names.REGRESSION:
                untested_df.sort_values(
                    by=[self.predictions_col], ascending=False, inplace=True)
            elif self.run_type == Names.CLASSIFICATION:
                # assuming binary classification, predictions of class 1 are ranked higher than class 0,
                # and the probability of a sample being in class 1 is used as the secondary column for ranking.
                # currently the _predict_proba methods in test harness model classes return the probability of a sample being in class 1
                untested_df.sort_values(by=[self.predictions_col, self.prob_predictions_col], ascending=[
                                        False, False], inplace=True)
            else:
                raise ValueError("self.run_type must be {} or {}".format(
                    Names.REGRESSION, Names.CLASSIFICATION))
            # resetting index to match sorted values, so the index can be used as a ranking.
            untested_df.reset_index(inplace=True, drop=True)
            # adding 1 to rankings so they start from 1 instead of 0.
            untested_df[self.rankings_col] = untested_df.index + 1

            print(("Prediction time of untested data was: {}".format(
                time.time() - prediction_start_time)))
            # Saving untested predictions
            self.untested_data_predictions = untested_df.copy()
        else:
            self.untested_data_predictions = None

    def calculate_metrics(self):
        self.metrics_dict[Names.NUM_FEATURES_USED] = len(
            self.feature_cols_to_use)
        if self.feature_cols_to_normalize:
            self.metrics_dict[Names.NUM_FEATURES_NORMALIZED] = len(
                self.feature_cols_to_normalize)
        else:
            self.metrics_dict[Names.NUM_FEATURES_NORMALIZED] = 0
        self.metrics_dict[Names.SAMPLES_IN_TRAIN] = len(self.training_data)
        self.metrics_dict[Names.SAMPLES_IN_TEST] = len(
            self.testing_data_predictions)

        if self.run_type == Names.CLASSIFICATION:
            self.metrics_dict[Names.NUM_CLASSES] = self.num_classes

            # this if/else block is needed for f1 score, precision, and recall
            if self.multiclass:
                averaging_type = "weighted"
            else:
                averaging_type = "binary"

            # the try/except blocks will allow AUC and Average Precision to be filled in with NaN if they can't be calculated
            # this removes the need for special logic to check if self.multiclass is True or False
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
                                                         self.testing_data_predictions[self.predictions_col],
                                                         average=averaging_type)
            self.metrics_dict[Names.PRECISION] = precision_score(self.testing_data_predictions[self.col_to_predict],
                                                                 self.testing_data_predictions[self.predictions_col],
                                                                 average=averaging_type)
            self.metrics_dict[Names.RECALL] = recall_score(self.testing_data_predictions[self.col_to_predict],
                                                           self.testing_data_predictions[self.predictions_col],
                                                           average=averaging_type)
        elif self.run_type == Names.REGRESSION:
            self.metrics_dict[Names.RMSE] = sqrt(
                mean_squared_error(self.testing_data_predictions[self.col_to_predict], self.testing_data_predictions[self.predictions_col]))
            self.metrics_dict[Names.R_SQUARED] = r2_score(self.testing_data_predictions[self.col_to_predict],
                                                          self.testing_data_predictions[self.predictions_col])
        else:
            raise TypeError("self.run_type must equal '{}' or '{}'".format(
                Names.CLASSIFICATION, Names.REGRESSION))

    # ---------------------------------------------------------
    # model on model
    def interpret_model(self,
                        complex_model,
                        training_df,
                        feature_col,
                        predict_col,
                        simple_model):
        """
        Trains an interpretable model on the predicted labels of 
        an uninterpretable model, thus offering an approximation of
        how the original model learned.

        * complex_model must already have its model parameters defined
        * simple_model must be defined, else it will be a default DecisionTreeClassifier
        """

        # train the complex model and get its predictions for training data
        complex_model.fit(training_df[feature_col], training_df[predict_col])
        predictions = complex_model.predict(training_df[feature_col])
        # check if predicted labels are continuous. If so, change to binary
        # TODO: modify this to allow for multilabel classification
        if len(np.unique(predictions)) > 2:
            predictions = [i[0] > .5 for i in predictions]

        # train simple model on predictions
        model_interpretation_img = self.get_simple_model_image(
            training_df[feature_col], predictions)
        self.model_interpretation_img = model_interpretation_img

    def get_simple_model_image(self, data_features, predicted_labels, interpretable_model=None):
        # make interpretable model a decision tree by default
        if interpretable_model == None:
            from sklearn.tree import DecisionTreeClassifier
            interpretable_model = DecisionTreeClassifier(random_state=0)

        # fit interpretable model on complex model's predicted labels
        # NOTE: random_state param has to be passed to get consistent tree output, but is an arbitrary number
        # How do we know which tree is the 'correct' way that the UninterpretableModel is learning?
        interpretable_model.fit(data_features, predicted_labels)

        # visualize model
        if str(type(interpretable_model)) == "<class 'sklearn.tree.tree.DecisionTreeClassifier'>":
            from sklearn.externals.six import StringIO
            #from PIL import Image
            from sklearn.tree import export_graphviz
            import pydotplus
            dot_data = StringIO()
            export_graphviz(interpretable_model, out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True,
                            feature_names=list(data_features.columns))
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

            # return graph.create_png()
            return dot_data
