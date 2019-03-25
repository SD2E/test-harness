import time
import rfpimp
import warnings
import pandas as pd
import numpy as np
from math import sqrt, fabs
from datetime import datetime
from sklearn import preprocessing
from eli5.sklearn import PermutationImportance
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, r2_score
from harness.unique_id import get_id
from harness.utils.names import Names
from harness.test_harness_models_abstract_classes import ClassificationModel, RegressionModel

import shap
import BlackBoxAuditing as BBA
from operator import itemgetter
from BlackBoxAuditing.model_factories.SKLearnModelVisitor import SKLearnModelVisitor
import matplotlib.pyplot as plt

plt.switch_backend('agg')


class _BaseRun:
    def __init__(self, test_harness_model, training_data, testing_data, data_and_split_description,
                 col_to_predict, feature_cols_to_use, index_cols, normalize, feature_cols_to_normalize,
                 feature_extraction, predict_untested_data=False, loo_dict=False):
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
        self.shap_values = None
        self.shap_plots_dict = None
        self.feature_importances = None

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

    # TODO: add different options for eli5.sklearn.permutation_importance (current usage) and eli5.permutation_importance
    def feature_extraction_method(self, method=Names.ELI5_PERMUTATION):
        print("Starting Feature Extraction...")
        start_time = time.time()

        if method == True:
            method = Names.ELI5_PERMUTATION

        if method == Names.ELI5_PERMUTATION:
            pi_object = PermutationImportance(self.test_harness_model.model)
            pi_object.fit(self.testing_data[self.feature_cols_to_use], self.testing_data[self.col_to_predict])
            feature_importances_df = pd.DataFrame()
            feature_importances_df["Feature"] = self.feature_cols_to_use
            feature_importances_df["Importance"] = pi_object.feature_importances_
            feature_importances_df["Importance_Std"] = pi_object.feature_importances_std_
            feature_importances_df.sort_values(by='Importance', inplace=True, ascending=False)
            self.feature_importances = feature_importances_df.copy()
        elif method == Names.RFPIMP_PERMUTATION:
            pis = rfpimp.importances(self.test_harness_model.model, self.testing_data[self.feature_cols_to_use],
                                     self.testing_data[self.col_to_predict])
            pis['Feature'] = pis.index
            pis.reset_index(inplace=True, drop=True)
            pis = pis[['Feature', 'Importance']]
            pis.sort_values(by='Importance', inplace=True, ascending=False)
            self.feature_importances = pis.copy()
        elif method == "sklearn_rf_default":
            pass  # TODO

        elif method == Names.BBA_AUDIT:
            data = self.perform_bba_audit(training_data=self.training_data.copy(),
                                          testing_data=self.testing_data.copy(),
                                          features=self.feature_cols_to_use,
                                          classifier=self.test_harness_model.model,
                                          col_to_predict=self.col_to_predict)
            feature_importances_df = pd.DataFrame(data, columns=["Feature", "Importance"])
            self.feature_importances = feature_importances_df.copy()

        elif method == Names.SHAP_AUDIT:
            self.shap_plots_dict = {}
            data = self.perform_shap_audit()
            feature_importances_df = pd.DataFrame(data, columns=["Feature", "Importance"])
            self.feature_importances = feature_importances_df.copy()

        print(("Feature Extraction time with method {0} was: {1:.2f} seconds".format(method, time.time() - start_time)))

    def perform_shap_audit(self):
        """
        From:
        https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20scikit-learn.html
        """
        import warnings
        warnings.filterwarnings('ignore')

        features = self.feature_cols_to_use
        classifier = self.test_harness_model
        train_X = self.training_data[features]
        test_X = self.testing_data[features]

        shap.initjs()
        # f = lambda x: classifier._predict_proba(x)[:,1]
        f = classifier._predict_proba
        train_X_df = pd.DataFrame(data=train_X, columns=features)
        # train_X_df = self.training_data.copy()
        med = train_X_df.median().values.reshape((1, train_X_df.shape[1]))
        explainer = shap.KernelExplainer(f, med)
        test_X_df = pd.DataFrame(data=test_X, columns=features)
        # test_X_df = self.testing_data.copy()
        shap_values = explainer.shap_values(test_X_df)

        # store shap_values so they can be accessed and output by TestHarness class
        self.shap_values = self.training_data[self.index_cols + features].copy()
        self.shap_values[features] = shap_values.copy()

        means = []
        totals_list = [0.0 for f in features]
        for val_list in shap_values:
            for i in range(0, len(val_list)):
                totals_list[i] += fabs(val_list[i])
        means = [(feat, total / len(shap_values)) for feat, total in zip(features, totals_list)]
        # mean_shaps are returned for use as feature_importances
        mean_shaps = sorted(means, key=itemgetter(1), reverse=True)

        # generating plots
        print("Generating SHAP plots!")

        plt.close("all")  # close all previous pyplot figures

        # Base Values vs. Model Output for first prediction
        shap.force_plot(explainer.expected_value, shap_values[0, :], test_X_df.iloc[0, :], matplotlib=True, show=False)
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["example_single_sample_force_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        # TODO: figure out how to save multiple sample force plots as variable
        # # Base Values vs. Model Output for full predictions
        # shap.force_plot(explainer.expected_value, shap_values, train_X_df, matplotlib=True, show=False)
        # fig = plt.gcf()      # get current figure
        # self.shap_plots_dict["force_plot_all_samples"] = fig
        # plt.close("all")     # close all previous pyplot figures

        # Dependencies plots
        for feature in self.feature_cols_to_use:
            shap.dependence_plot(feature, shap_values, test_X_df, interaction_index='auto', show=False)
            fig = plt.gcf()  # get current figure
            self.shap_plots_dict["dependence_plot_{}".format(feature.upper())] = fig
            plt.close("all")  # close all previous pyplot figures

        # Feature importance summaries in 3 plots: dot, violin, and bar
        shap.summary_plot(shap_values, test_X_df, show=False, plot_type="dot")
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["summary_dot_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        shap.summary_plot(shap_values, test_X_df, show=False, plot_type="violin")
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["summary_violin_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        shap.summary_plot(shap_values, test_X_df, plot_type="bar", show=False)
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["summary_bar_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        return mean_shaps

    def perform_bba_audit(self, training_data,
                          testing_data,
                          features,
                          classifier,
                          col_to_predict):
        combined_df = training_data.append(testing_data)
        X = combined_df[features]
        y = pd.DataFrame(combined_df[col_to_predict], columns=[col_to_predict])

        data = BBA.data.load_testdf_only(X, y)
        response_index = len(data[0]) - 1
        auditor = BBA.Auditor()
        auditor.trained_model = SKLearnModelVisitor(classifier, response_index)
        auditor(data)
        print("BBA AUDITOR RESULTS:\n")
        print(auditor._audits_data["ranks"])
        return auditor._audits_data["ranks"]

    def train_and_test_model(self):
        if self.normalize is not False:
            self._normalize_dataframes()

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
